module JAIDE.Core.Types where

open import Data.Nat using (ℕ; _≥?_)
open import Data.Vec using (Vec; []; _∷_; lookup; take; drop)
open import Data.Float using (Float)
open import Data.Bool using (Bool; true; false)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Data.Product using (Σ; _,_)

record Layer (in out : ℕ) : Set where
  field
    weights : Vec (Vec Float in) out

record SecureMessage (len : ℕ) : Set where
  field
    payload : Vec ℕ len

parseMessage : ∀ {n} → Vec ℕ n → Σ ℕ (λ len → SecureMessage len)
parseMessage {n} buffer with n ≥? 16
... | true = let len = lookup buffer 8
                 payload = take len (drop 16 buffer)
             in (len , record { payload = payload })
... | false = (0 , record { payload = [] })

GenesisLayer : Set
GenesisLayer = Layer 16 4
