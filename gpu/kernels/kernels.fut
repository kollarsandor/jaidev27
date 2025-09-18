entry add_arrays (a: []f32) (b: []f32) : []f32 =
  map2 (+) a b

entry matmul (a: [][]f32) (b: [][]f32) : [][]f32 =
  map (\row -> map (\col -> f32.sum (map2 (*) row col)) (transpose b)) a

entry quantum_correlation (n: i32) (shots: i32) (theta: f32) : f32 =
  let size = 1 << n
  in let probs = tabulate size (\i -> f32.sin(theta * f32.i32 i))
     in f32.sum probs / f32.i32 shots

entry tgn_memory_update (memory: []f32) (nodes: []f32) (time_factor: f32) : []f32 =
  map2 (\m n -> m * (1 - time_factor) + n * time_factor) memory nodes

entry spectral_radius_regulate (matrix: [][]f32) (target: f32) : [][]f32 =
  let dim = length matrix
  let matvec m v = map (\row -> f32.sum (map2 (*) row v)) m
  let dot u v = f32.sum (map2 (*) u v)
  let norm2 v = f32.sqrt (dot v v)
  let normalize v = map (\x -> x / norm2 v) v
  let v0 = replicate dim 1.0
  let v = loop v' = normalize v0 for _i < 46 do
            let av = matvec matrix v'
            normalize av
  let av = matvec matrix v
  let rho = dot v av / dot v v
  in map (\row -> map (\x -> x * (target / rho)) row) matrix
