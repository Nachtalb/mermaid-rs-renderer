//! Experimental orthogonal router: one A* per edge on a sparse grid, then nudging.
//!
//! The grid lines are the node borders, node centres and a margin around every
//! node (a Hanan-style grid, a few hundred to a few thousand vertices). Grid
//! edges inside or along a non-endpoint node are blocked, so every route avoids
//! nodes by construction and no sampling or collision scoring is needed. The
//! search state includes the travel direction so bends are costed exactly.
//! Edges are routed shortest-first; grid edges already used by earlier routes
//! get a penalty (soft sharing) and so do perpendicular passes over them
//! (crossings). Afterwards, collinear overlapping segments are spread onto
//! parallel lanes by greedy interval colouring.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use crate::config::LayoutConfig;
use crate::ir::{Direction, Graph};

use super::super::{EdgeLayout, NodeLayout, SubgraphLayout, TextBlock};
use super::edge_pipeline::effective_edge_endpoint_layouts;
use super::post_route;

const MARGIN: f32 = 14.0;
const LANE: f32 = 7.0;

fn knob(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Rect {
    fn of(n: &NodeLayout) -> Self {
        Rect {
            x0: n.x,
            y0: n.y,
            x1: n.x + n.width,
            y1: n.y + n.height,
        }
    }
    fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }
    fn contains(&self, p: (f32, f32)) -> bool {
        p.0 > self.x0 && p.0 < self.x1 && p.1 > self.y0 && p.1 < self.y1
    }
    /// Admissible lower bound on the remaining cost when standing at `p` heading `d`:
    /// L1 distance plus the minimum number of bends still needed to enter the box.
    fn h(&self, p: (f32, f32), d: usize, bend: f32) -> f32 {
        let in_x = p.0 > self.x0 && p.0 < self.x1;
        let in_y = p.1 > self.y0 && p.1 < self.y1;
        let away = match d {
            0 => p.0 >= self.x1,
            1 => p.0 <= self.x0,
            2 => p.1 >= self.y1,
            _ => p.1 <= self.y0,
        };
        let horizontal = d < 2;
        let bends = if away {
            2.0
        } else if (!in_x && !in_y) || (in_x && horizontal) || (in_y && !horizontal) {
            1.0
        } else {
            0.0
        };
        self.l1_dist(p) + bends * bend
    }
    fn l1_dist(&self, p: (f32, f32)) -> f32 {
        (self.x0 - p.0).max(0.0).max(p.0 - self.x1) + (self.y0 - p.1).max(0.0).max(p.1 - self.y1)
    }
}

#[derive(Default, Debug)]
pub(in crate::layout) struct GridRouterStats {
    pub routed: usize,
    pub failed: usize,
    pub vertices: usize,
    pub expanded: usize,
    pub node_hits: usize,
}

// Directions: 0 = +x, 1 = -x, 2 = +y, 3 = -y.
const DX: [isize; 4] = [1, -1, 0, 0];
const DY: [isize; 4] = [0, 0, 1, -1];

fn dedup_sorted(mut v: Vec<f32>) -> Vec<f32> {
    v.sort_by(f32::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    v
}

fn idx_of(v: &[f32], x: f32) -> usize {
    match v.binary_search_by(|p| p.total_cmp(&x)) {
        Ok(i) => i,
        Err(i) => {
            if i > 0 && (i == v.len() || (x - v[i - 1]).abs() <= (v[i] - x).abs()) {
                i - 1
            } else {
                i
            }
        }
    }
}

struct Grid {
    xs: Vec<f32>,
    ys: Vec<f32>,
    /// Blocked grid edge from vertex `v` towards +x (h) or +y (v).
    h_block: Vec<bool>,
    v_block: Vec<bool>,
    h_used: Vec<u16>,
    v_used: Vec<u16>,
}

impl Grid {
    fn nx(&self) -> usize {
        self.xs.len()
    }
    fn vid(&self, i: usize, j: usize) -> usize {
        j * self.nx() + i
    }
    fn pos(&self, v: usize) -> (f32, f32) {
        (self.xs[v % self.nx()], self.ys[v / self.nx()])
    }
    /// Grid-edge slot for a step from `v` in direction `d`: (is_horizontal, slot index).
    fn slot(&self, v: usize, d: usize) -> (bool, usize) {
        let nx = self.nx();
        match d {
            0 => (true, v),
            1 => (true, v - 1),
            2 => (false, v),
            _ => (false, v - nx),
        }
    }
    fn step(&self, v: usize, d: usize) -> Option<usize> {
        let (i, j) = ((v % self.nx()) as isize, (v / self.nx()) as isize);
        let (ni, nj) = (i + DX[d], j + DY[d]);
        if ni < 0 || nj < 0 || ni as usize >= self.nx() || nj as usize >= self.ys.len() {
            return None;
        }
        let (h, s) = self.slot(v, d);
        let blocked = if h { self.h_block[s] } else { self.v_block[s] };
        (!blocked).then(|| self.vid(ni as usize, nj as usize))
    }
    fn block_rect(&mut self, r: &Rect) {
        let (ia, ib) = (idx_of(&self.xs, r.x0), idx_of(&self.xs, r.x1));
        let (ja, jb) = (idx_of(&self.ys, r.y0), idx_of(&self.ys, r.y1));
        for j in ja..=jb {
            for i in ia..ib {
                let v = self.vid(i, j);
                self.h_block[v] = true;
            }
        }
        for i in ia..=ib {
            for j in ja..jb {
                let v = self.vid(i, j);
                self.v_block[v] = true;
            }
        }
    }
}

/// Port candidates on a box: (grid vertex, direction leaving the box, side penalty).
fn ports(
    grid: &Grid,
    r: &Rect,
    toward: (f32, f32),
    dir: Direction,
    bend: f32,
) -> Vec<(usize, usize, f32)> {
    let c = r.center();
    let (dx, dy) = (toward.0 - c.0, toward.1 - c.1);
    let vertical_flow = matches!(dir, Direction::TopDown | Direction::BottomTop);
    // Preferred side: along the flow axis when the other end is clearly ahead/behind.
    let pref = if vertical_flow && dy.abs() > r.y1 - r.y0 {
        if dy > 0.0 { 2 } else { 3 }
    } else if !vertical_flow && dx.abs() > r.x1 - r.x0 {
        if dx > 0.0 { 0 } else { 1 }
    } else if dx.abs() * (r.y1 - r.y0) > dy.abs() * (r.x1 - r.x0) {
        if dx > 0.0 { 0 } else { 1 }
    } else if dy > 0.0 {
        2
    } else {
        3
    };
    let opposite = [1, 0, 3, 2];
    let pts = [(r.x1, c.1), (r.x0, c.1), (c.0, r.y1), (c.0, r.y0)];
    (0..4)
        .map(|d| {
            let p = pts[d];
            let v = grid.vid(idx_of(&grid.xs, p.0), idx_of(&grid.ys, p.1));
            let pen = if d == pref {
                0.0
            } else if d == opposite[pref] {
                3.0 * bend
            } else {
                bend
            };
            (v, d, pen)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(in crate::layout) fn build_grid_routed_edges(
    graph: &Graph,
    nodes: &BTreeMap<String, NodeLayout>,
    subgraphs: &[SubgraphLayout],
    config: &LayoutConfig,
    edge_route_labels: &[Option<TextBlock>],
    edge_start_labels: &[Option<TextBlock>],
    edge_end_labels: &[Option<TextBlock>],
    stats: &mut GridRouterStats,
) -> Vec<EdgeLayout> {
    let bend = knob("MMDR_G_BEND", 28.0);
    let share = knob("MMDR_G_SHARE", 0.5); // extra cost per px on an already used grid edge
    let cross = knob("MMDR_G_CROSS", 300.0);
    let subgraph_cost = knob("MMDR_G_SUB", 120.0);
    let greed = knob("MMDR_G_GREED", 1.0); // heuristic weight; >1 = weighted A*
    let obstacles: Vec<(&str, Rect)> = nodes
        .iter()
        .filter(|(_, n)| !n.hidden && n.anchor_subgraph.is_none())
        .map(|(id, n)| (id.as_str(), Rect::of(n)))
        .collect();
    // Edges to a subgraph attach to the subgraph's box, not its tiny hidden anchor node.
    let sub_rect = |id: &str| {
        let k = nodes.get(id)?.anchor_subgraph?;
        let s = subgraphs.get(k)?;
        Some(Rect {
            x0: s.x,
            y0: s.y,
            x1: s.x + s.width,
            y1: s.y + s.height,
        })
    };
    let ends: Vec<Option<(Rect, Rect)>> = graph
        .edges
        .iter()
        .map(|e| {
            let (f, t) = effective_edge_endpoint_layouts(graph, nodes, subgraphs, e)?;
            Some((
                sub_rect(&e.from).unwrap_or(Rect::of(&f)),
                sub_rect(&e.to).unwrap_or(Rect::of(&t)),
            ))
        })
        .collect();

    // Grid lines.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let all_boxes = obstacles
        .iter()
        .map(|(_, r)| *r)
        .chain(ends.iter().flatten().flat_map(|(a, b)| [*a, *b]));
    let (mut lo, mut hi) = ((f32::MAX, f32::MAX), (f32::MIN, f32::MIN));
    for r in all_boxes {
        let c = r.center();
        xs.extend([r.x0 - MARGIN, r.x0, c.0, r.x1, r.x1 + MARGIN]);
        ys.extend([r.y0 - MARGIN, r.y0, c.1, r.y1, r.y1 + MARGIN]);
        lo = (lo.0.min(r.x0), lo.1.min(r.y0));
        hi = (hi.0.max(r.x1), hi.1.max(r.y1));
    }
    for s in subgraphs {
        xs.extend([s.x - MARGIN / 2.0, s.x + s.width + MARGIN / 2.0]);
        ys.extend([s.y - MARGIN / 2.0, s.y + s.height + MARGIN / 2.0]);
    }
    xs.extend([lo.0 - 3.0 * MARGIN, hi.0 + 3.0 * MARGIN]);
    ys.extend([lo.1 - 3.0 * MARGIN, hi.1 + 3.0 * MARGIN]);
    // Channel centres between neighbouring lines give routes a tidy middle lane.
    let mid = |v: Vec<f32>| {
        let v = dedup_sorted(v);
        let mut out = v.clone();
        out.extend(
            v.windows(2)
                .filter(|w| w[1] - w[0] > 3.0 * MARGIN)
                .map(|w| (w[0] + w[1]) / 2.0),
        );
        dedup_sorted(out)
    };
    let (xs, ys) = (mid(xs), mid(ys));
    let nv = xs.len() * ys.len();
    let mut grid = Grid {
        xs,
        ys,
        h_block: vec![false; nv],
        v_block: vec![false; nv],
        h_used: vec![0; nv],
        v_used: vec![0; nv],
    };
    for (_, r) in &obstacles {
        grid.block_rect(r);
    }
    stats.vertices = nv;
    // Bitmask of subgraphs containing each grid vertex (first 64 subgraphs).
    let sub_rects: Vec<Rect> = subgraphs
        .iter()
        .take(64)
        .map(|s| Rect {
            x0: s.x,
            y0: s.y,
            x1: s.x + s.width,
            y1: s.y + s.height,
        })
        .collect();
    let sub_mask: Vec<u64> = (0..nv)
        .map(|v| {
            let p = grid.pos(v);
            sub_rects
                .iter()
                .enumerate()
                .filter(|(_, r)| r.contains(p))
                .fold(0, |m, (k, _)| m | (1 << k))
        })
        .collect();

    // Route shortest edges first so they get the straight lanes.
    let mut order: Vec<usize> = (0..graph.edges.len())
        .filter(|&i| ends[i].is_some())
        .collect();
    order.sort_by(|&a, &b| {
        let len = |i: usize| {
            let (f, t) = ends[i].unwrap();
            let (p, q) = (f.center(), t.center());
            (p.0 - q.0).abs() + (p.1 - q.1).abs()
        };
        len(a).total_cmp(&len(b))
    });

    let mut dist = vec![f32::INFINITY; nv * 4];
    let mut prev = vec![u32::MAX; nv * 4];
    let mut touched: Vec<usize> = Vec::new();
    let mut routed: Vec<Vec<(f32, f32)>> = vec![Vec::new(); graph.edges.len()];
    for i in order {
        let e = &graph.edges[i];
        let (fr, tr) = ends[i].unwrap();
        if e.from == e.to {
            let (cx, cy) = fr.center();
            let w = ((fr.y1 - fr.y0) / 4.0).max(6.0);
            routed[i] = vec![
                (fr.x1, cy - w),
                (fr.x1 + 2.0 * MARGIN, cy - w),
                (fr.x1 + 2.0 * MARGIN, cy + w),
                (fr.x1, cy + w),
            ];
            let _ = cx;
            stats.routed += 1;
            continue;
        }
        // Foreign subgraphs: those containing exactly one endpoint, or neither.
        // Foreign subgraphs: those containing neither endpoint.
        let foreign: u64 = sub_rects
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.contains(fr.center()) && !s.contains(tr.center()))
            .fold(0, |m, (k, _)| m | (1 << k));
        for &k in &touched {
            dist[k] = f32::INFINITY;
            prev[k] = u32::MAX;
        }
        touched.clear();
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for (v, d, pen) in ports(&grid, &fr, tr.center(), graph.direction, bend) {
            let s = v * 4 + d;
            if pen < dist[s] {
                dist[s] = pen;
                touched.push(s);
                heap.push(Reverse((
                    (pen + greed * tr.h(grid.pos(v), d, bend)).to_bits(),
                    s as u32,
                )));
            }
        }
        // Goal: arrive at a target port moving into the box.
        let goals: Vec<(usize, usize, f32)> = ports(&grid, &tr, fr.center(), graph.direction, bend)
            .into_iter()
            .map(|(v, d, pen)| (v, [1, 0, 3, 2][d], pen))
            .collect();
        let mut best: Option<(f32, usize)> = None;
        while let Some(Reverse((fbits, s))) = heap.pop() {
            let s = s as usize;
            let f = f32::from_bits(fbits);
            if best.is_some_and(|(b, _)| f >= b) {
                break;
            }
            let (v, d) = (s / 4, s % 4);
            let g = dist[s];
            if f > g + greed * tr.h(grid.pos(v), d, bend) + 0.01 {
                continue; // stale
            }
            stats.expanded += 1;
            if let Some(&(_, _, pen)) = goals.iter().find(|&&(gv, gd, _)| gv == v && gd == d) {
                let total = g + pen;
                if best.is_none_or(|(b, _)| total < b) {
                    best = Some((total, s));
                }
            }
            let p = grid.pos(v);
            for nd in 0..4 {
                if nd == [1, 0, 3, 2][d] {
                    continue; // no U-turns
                }
                let Some(nv_) = grid.step(v, nd) else {
                    continue;
                };
                let q = grid.pos(nv_);
                let len = (q.0 - p.0).abs() + (q.1 - p.1).abs();
                let (h, slot) = grid.slot(v, nd);
                let used = if h {
                    grid.h_used[slot]
                } else {
                    grid.v_used[slot]
                };
                let mut cost = len * (1.0 + share * used as f32);
                if nd != d {
                    cost += bend;
                }
                // Crossing: passing straight through a vertex used perpendicular to us.
                let perp_used = |w: usize| {
                    let (a, b) = if h {
                        (
                            w.checked_sub(grid.nx()).map(|x| grid.v_used[x]),
                            Some(grid.v_used[w]),
                        )
                    } else {
                        (
                            w.checked_sub(1).map(|x| grid.h_used[x]),
                            Some(grid.h_used[w]),
                        )
                    };
                    a.unwrap_or(0) > 0 && b.unwrap_or(0) > 0
                };
                if perp_used(nv_) {
                    cost += cross;
                }
                cost +=
                    subgraph_cost * ((sub_mask[v] ^ sub_mask[nv_]) & foreign).count_ones() as f32;
                let ns = nv_ * 4 + nd;
                let ng = g + cost;
                if ng < dist[ns] {
                    if dist[ns].is_infinite() {
                        touched.push(ns);
                    }
                    dist[ns] = ng;
                    prev[ns] = s as u32;
                    heap.push(Reverse((
                        (ng + greed * tr.h(q, nd, bend)).to_bits(),
                        ns as u32,
                    )));
                }
            }
        }
        let Some((_, mut s)) = best else {
            stats.failed += 1;
            if std::env::var_os("MMDR_ROUTER_STATS").is_some() {
                eprintln!("grid_router: no path {} -> {}", e.from, e.to);
            }
            let (a, b) = (fr.center(), tr.center());
            routed[i] = vec![a, (a.0, (a.1 + b.1) / 2.0), (b.0, (a.1 + b.1) / 2.0), b];
            continue;
        };
        let mut verts = vec![s / 4];
        while prev[s] != u32::MAX {
            let p = prev[s] as usize;
            let (h, slot) = grid.slot(p / 4, s % 4);
            if h {
                grid.h_used[slot] += 1;
            } else {
                grid.v_used[slot] += 1;
            }
            s = p;
            verts.push(s / 4);
        }
        verts.reverse();
        let mut pts: Vec<(f32, f32)> = verts.into_iter().map(|v| grid.pos(v)).collect();
        simplify(&mut pts);
        routed[i] = pts;
        stats.routed += 1;
    }

    nudge(&mut routed);

    for (i, pts) in routed.iter().enumerate() {
        let Some(e) = graph.edges.get(i) else {
            continue;
        };
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let m = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            if obstacles
                .iter()
                .any(|(id, r)| *id != e.from && *id != e.to && seg_hits(r, a, b, m))
            {
                stats.node_hits += 1;
            }
        }
    }

    let label_anchors: Vec<Option<(f32, f32)>> = routed
        .iter()
        .map(|pts| {
            pts.windows(2)
                .max_by(|a, b| seg_len(a).total_cmp(&seg_len(b)))
                .map(|w| ((w[0].0 + w[1].0) / 2.0, (w[0].1 + w[1].1) / 2.0))
        })
        .collect();
    post_route::build_edge_layouts(
        graph,
        &routed,
        edge_route_labels,
        edge_start_labels,
        edge_end_labels,
        &label_anchors,
        config,
    )
}

fn seg_len(w: &[(f32, f32)]) -> f32 {
    (w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs()
}

fn seg_hits(r: &Rect, a: (f32, f32), b: (f32, f32), _m: (f32, f32)) -> bool {
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    x1 > r.x0 + 0.5 && x0 < r.x1 - 0.5 && y1 > r.y0 + 0.5 && y0 < r.y1 - 0.5
}

/// Drop collinear interior points.
fn simplify(pts: &mut Vec<(f32, f32)>) {
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    for &p in pts.iter() {
        if out.len() >= 2 {
            let a = out[out.len() - 2];
            let b = out[out.len() - 1];
            if ((a.0 - b.0).abs() < 0.01 && (b.0 - p.0).abs() < 0.01)
                || ((a.1 - b.1).abs() < 0.01 && (b.1 - p.1).abs() < 0.01)
            {
                out.pop();
            }
        }
        out.push(p);
    }
    *pts = out;
}

/// Spread collinear overlapping segments of different edges onto parallel lanes.
fn nudge(routed: &mut [Vec<(f32, f32)>]) {
    // (horizontal?, coord key) -> [(lo, hi, edge, seg)]
    let mut groups: BTreeMap<(bool, i32), Vec<(f32, f32, usize, usize)>> = BTreeMap::new();
    for (e, pts) in routed.iter().enumerate() {
        for k in 0..pts.len().saturating_sub(1) {
            let (a, b) = (pts[k], pts[k + 1]);
            let h = (a.1 - b.1).abs() < 0.01;
            let (c, lo, hi) = if h {
                (a.1, a.0.min(b.0), a.0.max(b.0))
            } else {
                (a.0, a.1.min(b.1), a.1.max(b.1))
            };
            groups
                .entry((h, c.round() as i32))
                .or_default()
                .push((lo, hi, e, k));
        }
    }
    for ((h, _), mut segs) in groups {
        if segs.len() < 2 {
            continue;
        }
        segs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut colored: Vec<(f32, f32, usize, usize, usize)> = Vec::new();
        for (lo, hi, e, k) in segs {
            let mut c = 0;
            while colored
                .iter()
                .any(|&(l2, h2, e2, _, c2)| c2 == c && e2 != e && hi > l2 + 0.5 && lo < h2 - 0.5)
            {
                c += 1;
            }
            colored.push((lo, hi, e, k, c));
        }
        let n = colored.iter().map(|x| x.4).max().unwrap_or(0) + 1;
        if n < 2 {
            continue;
        }
        for (_, _, e, k, c) in colored {
            let off = (c as f32 - (n - 1) as f32 / 2.0) * LANE;
            let pts = &mut routed[e];
            for p in [k, k + 1] {
                if h {
                    pts[p].1 += off;
                } else {
                    pts[p].0 += off;
                }
            }
        }
    }
}
