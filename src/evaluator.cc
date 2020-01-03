/*
 *  Copyright 2016-2019 Henry Lee
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <phevaluator/phevaluator.h>
#include "hash.h"
#include "tables.h"

namespace phevaluator {

Rank EvaluateCards(const Card& a, const Card& b, const Card& c, const Card& d,
                   const Card& e) {
  return evaluate_5cards(a, b, c, d, e);
}

Rank EvaluateCards(const Card& a, const Card& b, const Card& c, const Card& d,
                   const Card& e, const Card& f) {
  return evaluate_6cards(a, b, c, d, e, f);
}

Rank EvaluateCards(const Card& a, const Card& b, const Card& c, const Card& d,
                   const Card& e, const Card& f, const Card& g) {
  return evaluate_7cards(a, b, c, d, e, f, g);
}

Rank EvaluateCards(const Card& a, const Card& b, const Card& c, const Card& d,
                   const Card& e, const Card& f, const Card& g, const Card& h) {
  return evaluate_8cards(a, b, c, d, e, f, g, h);
}

Rank EvaluateCards(const Card& a, const Card& b, const Card& c, const Card& d,
                   const Card& e, const Card& f, const Card& g, const Card& h,
                   const Card& i) {
  return evaluate_9cards(a, b, c, d, e, f, g, h, i);
}

Rank EvaluateHand(const Hand& hand) {
  if (suits[hand.getSuitHash()])
    return flush[hand.getSuitBinary()[suits[hand.getSuitHash()]-1]];

  const int hash = hash_quinary(hand.getQuinary().data(), 13, hand.size());

  switch (hand.size()) {
    case 5: return noflush5[hash];
    case 6: return noflush6[hash];
    case 7: return noflush7[hash];
    case 8: return noflush8[hash];
    case 9: return noflush9[hash];
  }

  return noflush5[hash];
}

} // namespace phevaluator

