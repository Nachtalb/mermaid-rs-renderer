//! Experimental layered orthogonal edge router (track assignment per layer gap).
//!
//! Instead of searching 2-D paths around obstacles, every edge between two
//! layers is drawn as at most three segments: out of the source along the rank
//! axis, across on a horizontal "track" inside one inter-layer gap, then into
//! the target. Choosing the tracks is colouring a directional interval graph:
//! intervals that contain each other need different tracks, and same-direction
//! intervals that partially overlap get an order so they cross at most once
//! (Gutowski et al., "Coloring Mixed and Directional Interval Graphs", GD 2022).
//! The greedy colouring is optimal per direction; left- and right-going edges
//! are stacked, giving at most 2x the minimum number of tracks.
//!
//! Prototype: node coordinates are taken as-is (gaps are not resized to fit the
//! tracks), and edges that cannot be drawn this way fall back to a plain Z.

use std::collections::BTreeMap;

use crate::config::LayoutConfig;
use crate::ir::{Direction, Graph};

use super::super::{EdgeLayout, NodeLayout, SubgraphLayout, TextBlock};
use super::edge_pipeline::effective_edge_endpoint_layouts;
use super::post_route;

/// Node box in rank-aligned coordinates: `u` is the cross axis, `v` the rank axis.
#[derive(Clone, Copy, Debug)]
struct Rect {
    u0: f32,
    u1: f32,
    v0: f32,
    v1: f32,
}

#[derive(Default, Debug)]
pub(in crate::layout) struct TrackRouterStats {
    pub routed: usize,
    pub fallback: usize,
    pub tracks: usize,
}

fn to_rect(n: &NodeLayout, dir: Direction) -> Rect {
    match dir {
        Direction::TopDown | Direction::BottomTop => Rect {
            u0: n.x,
            u1: n.x + n.width,
            v0: n.y,
            v1: n.y + n.height,
        },
        Direction::LeftRight | Direction::RightLeft => Rect {
            u0: n.y,
            u1: n.y + n.height,
            v0: n.x,
            v1: n.x + n.width,
        },
    }
}

fn from_uv(p: (f32, f32), dir: Direction) -> (f32, f32) {
    match dir {
        Direction::TopDown | Direction::BottomTop => p,
        Direction::LeftRight | Direction::RightLeft => (p.1, p.0),
    }
}

/// Group node boxes into layers by overlapping rank-axis extent.
fn layer_bands(rects: &[Rect]) -> Vec<(f32, f32)> {
    let mut spans: Vec<(f32, f32)> = rects.iter().map(|r| (r.v0, r.v1)).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut bands: Vec<(f32, f32)> = Vec::new();
    for (a, b) in spans {
        match bands.last_mut() {
            Some(last) if a < last.1 - 0.5 => last.1 = last.1.max(b),
            _ => bands.push((a, b)),
        }
    }
    bands
}

fn band_of(bands: &[(f32, f32)], r: &Rect) -> usize {
    let c = (r.v0 + r.v1) / 2.0;
    bands
        .iter()
        .position(|&(a, b)| c >= a - 0.5 && c <= b + 0.5)
        .unwrap_or(0)
}

/// Does a rank-axis segment at cross position `u` from `va` to `vb` pass through
/// any node other than the given endpoints?
fn vertical_blocked(rects: &[Rect], skip: [usize; 2], u: f32, va: f32, vb: f32) -> bool {
    let (lo, hi) = if va < vb { (va, vb) } else { (vb, va) };
    rects.iter().enumerate().any(|(i, r)| {
        !skip.contains(&i) && u > r.u0 + 0.5 && u < r.u1 - 0.5 && hi > r.v0 + 0.5 && lo < r.v1 - 0.5
    })
}

fn horizontal_blocked(rects: &[Rect], v: f32, ua: f32, ub: f32) -> bool {
    let (lo, hi) = if ua < ub { (ua, ub) } else { (ub, ua) };
    rects
        .iter()
        .any(|r| v > r.v0 + 0.5 && v < r.v1 - 0.5 && hi > r.u0 + 0.5 && lo < r.u1 - 0.5)
}

struct Planned {
    edge: usize,
    /// Port on the upper (smaller v) node and lower node.
    ua: f32,
    ub: f32,
    va: f32,
    vb: f32,
    gap: usize,
    reversed: bool,
}

/// Greedy colouring of one direction class. Intervals are `(left, right, id)`
/// with left-going semantics: if `a` starts left of `b` and they overlap
/// partially, `a` must get the smaller track. Containment only needs distinct
/// tracks. Returns track per id (0 = closest to the upper layer).
fn color_directional(mut iv: Vec<(f32, f32, usize)>) -> Vec<(usize, usize)> {
    iv.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut done: Vec<(f32, f32, usize)> = Vec::new(); // (l, r, color)
    let mut out = Vec::new();
    for (l, r, id) in iv {
        let mut min_c = 0usize;
        let mut forbidden: Vec<usize> = Vec::new();
        for &(pl, pr, pc) in &done {
            if pr <= l + 0.5 {
                continue; // disjoint
            }
            if pr < r - 0.5 {
                // partial overlap, previous is the left one
                min_c = min_c.max(pc + 1);
            } else {
                forbidden.push(pc); // containment
            }
            let _ = pl;
        }
        let mut c = min_c;
        while forbidden.contains(&c) {
            c += 1;
        }
        done.push((l, r, c));
        out.push((id, c));
    }
    out
}

pub(in crate::layout) fn build_track_routed_edges(
    graph: &Graph,
    nodes: &BTreeMap<String, NodeLayout>,
    subgraphs: &[SubgraphLayout],
    config: &LayoutConfig,
    edge_route_labels: &[Option<TextBlock>],
    edge_start_labels: &[Option<TextBlock>],
    edge_end_labels: &[Option<TextBlock>],
    stats: &mut TrackRouterStats,
) -> Vec<EdgeLayout> {
    let dir = graph.direction;
    let visible: Vec<(&String, Rect)> = nodes
        .iter()
        .filter(|(_, n)| !n.hidden)
        .map(|(id, n)| (id, to_rect(n, dir)))
        .collect();
    let index: BTreeMap<&str, usize> = visible
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id.as_str(), i))
        .collect();
    let rects: Vec<Rect> = visible.iter().map(|(_, r)| *r).collect();
    let bands = layer_bands(&rects);

    // Endpoint boxes (subgraph anchors resolved) per edge.
    let ends: Vec<Option<(Rect, Rect, usize, usize)>> = graph
        .edges
        .iter()
        .map(|e| {
            let (f, t) = effective_edge_endpoint_layouts(graph, nodes, subgraphs, e)?;
            let fi = *index.get(e.from.as_str()).unwrap_or(&usize::MAX);
            let ti = *index.get(e.to.as_str()).unwrap_or(&usize::MAX);
            Some((to_rect(&f, dir), to_rect(&t, dir), fi, ti))
        })
        .collect();

    // Port spreading: per (node, side) sort attached edges by the other end's u.
    // side: false = near side (v0), true = far side (v1)
    let mut side_users: BTreeMap<(String, bool), Vec<(f32, usize)>> = BTreeMap::new();
    for (i, e) in graph.edges.iter().enumerate() {
        let Some((f, t, ..)) = ends[i] else { continue };
        if e.from == e.to {
            continue;
        }
        let down = (f.v0 + f.v1) <= (t.v0 + t.v1);
        let (up_id, lo_id, up, lo) = if down {
            (&e.from, &e.to, f, t)
        } else {
            (&e.to, &e.from, t, f)
        };
        side_users
            .entry((up_id.clone(), true))
            .or_default()
            .push(((lo.u0 + lo.u1) / 2.0, i));
        side_users
            .entry((lo_id.clone(), false))
            .or_default()
            .push(((up.u0 + up.u1) / 2.0, i));
    }
    let mut port_u: BTreeMap<(usize, bool), f32> = BTreeMap::new(); // (edge, is_upper_end)
    for ((id, far), mut users) in side_users {
        let Some(n) = nodes.get(&id) else { continue };
        let r = to_rect(n, dir);
        users.sort_by(|a, b| a.0.total_cmp(&b.0));
        let k = users.len() as f32;
        for (j, (_, e)) in users.into_iter().enumerate() {
            let u = r.u0 + (r.u1 - r.u0) * (j as f32 + 1.0) / (k + 1.0);
            port_u.insert((e, far), u);
        }
    }

    let mut routed: Vec<Vec<(f32, f32)>> = vec![Vec::new(); graph.edges.len()];
    let mut planned: Vec<Planned> = Vec::new();
    for (i, e) in graph.edges.iter().enumerate() {
        let Some((f, t, fi, ti)) = ends[i] else {
            continue;
        };
        let down = (f.v0 + f.v1) <= (t.v0 + t.v1);
        let (up, lo, upi, loi) = if down { (f, t, fi, ti) } else { (t, f, ti, fi) };
        let bu = band_of(&bands, &up);
        let bl = band_of(&bands, &lo);
        let ua = port_u
            .get(&(i, true))
            .copied()
            .unwrap_or((up.u0 + up.u1) / 2.0);
        let ub = port_u
            .get(&(i, false))
            .copied()
            .unwrap_or((lo.u0 + lo.u1) / 2.0);
        let (va, vb) = (up.v1, lo.v0);
        let simple = e.from != e.to && bl > bu && vb > va;
        if simple {
            // Try the gap right above the target first (long vertical out of the
            // source), then the gap right below the source.
            let candidates = [bl - 1, bu];
            let skip = [upi, loi];
            let pick = candidates.into_iter().find(|&g| {
                let gv = (bands[g].1 + bands[g + 1].0) / 2.0;
                !vertical_blocked(&rects, skip, ua, va, gv)
                    && !vertical_blocked(&rects, skip, ub, gv, vb)
                    && !horizontal_blocked(&rects, gv, ua, ub)
            });
            if let Some(gap) = pick {
                planned.push(Planned {
                    edge: i,
                    ua,
                    ub,
                    va,
                    vb,
                    gap,
                    reversed: !down,
                });
                stats.routed += 1;
                continue;
            }
        }
        // Fallback: plain Z through the midpoint (no obstacle avoidance).
        stats.fallback += 1;
        if std::env::var_os("MMDR_TRACK_STATS").is_some() {
            let why = if e.from == e.to {
                "self-loop"
            } else if bl == bu {
                "same-layer"
            } else if vb <= va {
                "overlapping-ranks"
            } else {
                "blocked"
            };
            eprintln!(
                "  fallback {}->{}: {} (bands {}->{} of {})",
                e.from,
                e.to,
                why,
                bu,
                bl,
                bands.len()
            );
        }
        let (a, b) = ((ua, va), (ub, vb.max(va)));
        let mid = (a.1 + b.1) / 2.0;
        let mut pts = vec![a, (a.0, mid), (b.0, mid), b];
        if !down {
            pts.reverse();
        }
        routed[i] = pts.into_iter().map(|p| from_uv(p, dir)).collect();
    }

    // Track assignment per gap.
    let mut by_gap: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, p) in planned.iter().enumerate() {
        by_gap.entry(p.gap).or_default().push(k);
    }
    let mut track_v: Vec<f32> = vec![0.0; planned.len()];
    for (gap, ks) in by_gap {
        let (g0, g1) = (bands[gap].1, bands[gap + 1].0);
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut straight = Vec::new();
        for &k in &ks {
            let p = &planned[k];
            if (p.ua - p.ub).abs() < 0.5 {
                straight.push(k);
            } else if p.ub < p.ua {
                left.push((p.ub, p.ua, k));
            } else {
                // Right-going: mirror so the same left-going rule applies.
                right.push((-p.ub, -p.ua, k));
            }
        }
        let lc = color_directional(left);
        let rc = color_directional(right);
        let n_left = lc.iter().map(|&(_, c)| c + 1).max().unwrap_or(0);
        let n_right = rc.iter().map(|&(_, c)| c + 1).max().unwrap_or(0);
        let total = (n_left + n_right).max(1);
        stats.tracks += n_left + n_right;
        let step = (g1 - g0) / (total as f32 + 1.0);
        for (k, c) in lc {
            track_v[k] = g0 + step * (c as f32 + 1.0);
        }
        for (k, c) in rc {
            track_v[k] = g0 + step * ((n_left + c) as f32 + 1.0);
        }
        for k in straight {
            track_v[k] = (g0 + g1) / 2.0;
        }
    }

    let mut label_anchors: Vec<Option<(f32, f32)>> = vec![None; graph.edges.len()];
    for (k, p) in planned.iter().enumerate() {
        let tv = track_v[k];
        let mut pts = if (p.ua - p.ub).abs() < 0.5 {
            vec![(p.ua, p.va), (p.ub, p.vb)]
        } else {
            vec![(p.ua, p.va), (p.ua, tv), (p.ub, tv), (p.ub, p.vb)]
        };
        let anchor = if pts.len() == 4 {
            ((p.ua + p.ub) / 2.0, tv)
        } else {
            (p.ua, (p.va + p.vb) / 2.0)
        };
        label_anchors[p.edge] = Some(from_uv(anchor, dir));
        if p.reversed {
            pts.reverse();
        }
        routed[p.edge] = pts.into_iter().map(|p| from_uv(p, dir)).collect();
    }
    for (i, pts) in routed.iter().enumerate() {
        if label_anchors[i].is_none() && pts.len() >= 2 {
            let m = pts.len() / 2;
            let (a, b) = (pts[m - 1], pts[m]);
            label_anchors[i] = Some(((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0));
        }
    }

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

#[cfg(test)]
mod tests {
    use super::color_directional;

    #[test]
    fn partial_overlap_orders_left_interval_first() {
        // a=[0,10], b=[5,15] partially overlap: a must be on a smaller track.
        let c = color_directional(vec![(5.0, 15.0, 1), (0.0, 10.0, 0)]);
        let get = |id| c.iter().find(|x| x.0 == id).unwrap().1;
        assert!(get(0) < get(1));
    }

    #[test]
    fn containment_gets_distinct_tracks_and_disjoint_reuses() {
        let c = color_directional(vec![(0.0, 20.0, 0), (5.0, 10.0, 1), (30.0, 40.0, 2)]);
        let get = |id| c.iter().find(|x| x.0 == id).unwrap().1;
        assert_ne!(get(0), get(1));
        assert_eq!(get(2), 0);
    }
}
