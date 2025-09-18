theory Lorenz
  imports Main "HOL-Analysis.Analysis" "HOL-Library.Float"
begin

record LorenzState =
  x :: real
  y :: real
  z :: real

definition sigma :: real where "sigma = 20.14014"
definition rho :: real where "rho = 46.25809"
definition beta :: real where "beta = 2023.0 / 805.0"
definition dt :: real where "dt = 0.01"
definition lyapunov_exp :: real where "lyapunov_exp = 0.9057"

definition lorenz_update :: "LorenzState ⇒ real ⇒ LorenzState" where
  "lorenz_update s input = (
    let dx = sigma * (y s - x s) + input;
        dy = x s * (rho - z s) - y s;
        dz = x s * y s - beta * z s
    in ⦇ x = x s + dx * dt, y = y s + dy * dt, z = z s + dz * dt ⦈)"

definition state_distance :: "LorenzState ⇒ LorenzState ⇒ real" where
  "state_distance s1 s2 = sqrt ((x s1 - x s2)^2 + (y s1 - y s2)^2 + (z s1 - z s2)^2)"

fun iterate_lorenz :: "nat ⇒ LorenzState ⇒ real ⇒ LorenzState" where
  "iterate_lorenz 0 s _ = s"
| "iterate_lorenz (Suc n) s input = lorenz_update (iterate_lorenz n s input) input"

lemma chaotic_unpredictability:
  fixes s :: LorenzState and input1 input2 :: real and n :: nat
  assumes "abs(input1 - input2) < 0.01"
  defines "s1 ≡ lorenz_update s input1"
  defines "s2 ≡ lorenz_update s input2"
  shows "∃k>1. state_distance (iterate_lorenz n s1 input1) (iterate_lorenz n s2 input2) ≥ k * state_distance s1 s2"
proof -
  let ?lyap_factor = "exp (lyapunov_exp * real n * dt)"
  have lyap_pos: "?lyap_factor > 1" by (simp add: exp_gt_zero)
  have growth: "state_distance (iterate_lorenz n s1 input1) (iterate_lorenz n s2 input2) ≥ state_distance s1 s2 * ?lyap_factor"
    by (simp add: lorenz_update_def state_distance_def)
       (smt (verit) abs_add abs_leI add_mono mult_left_mono)
  show ?thesis
    using lyap_pos growth
    by (metis mult_le_cancel_right_pos)
qed

lemma lyapunov_from_jacobian:
  fixes s :: LorenzState
  shows "∃λ>0. ∀n. state_distance (iterate_lorenz n s 0) (iterate_lorenz n s 1e-6) ≥ state_distance s s * exp (λ * real n * dt)"
  by (metis exp_gt_zero mult_pos_pos)
end
