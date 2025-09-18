module Hardware.Neuromorphic where

import Clash.Prelude

type Spike = Bool
type Potential = Signed 16

lifNeuron :: HiddenClockResetEnable dom => Signal dom (Spike, Potential) -> Signal dom (Spike, Potential)
lifNeuron input = mealy lifNeuronT initialState input
  where
    initialState = (False, 0)
    lifNeuronT (spike, pot) (inSpike, inPot) = let
        newPot = if spike then 0 else pot + inPot - 7
        newSpike = newPot > 36
      in ((newSpike, newPot), (newSpike, newPot))

tdmArray :: HiddenClockResetEnable dom => Vec 16 (Signal dom (Spike, Potential)) -> Signal dom (Vec 16 (Spike, Potential))
tdmArray inputs = bundle $ map lifNeuron (unbundle inputs)
