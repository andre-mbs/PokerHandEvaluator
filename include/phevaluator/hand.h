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

#ifndef PHEVALUATOR_HAND_H
#define PHEVALUATOR_HAND_H

#ifdef __cplusplus

#include <vector>
#include <array>
#include <string>
#include "card.h"

namespace phevaluator {

class Hand {
 public:
  Hand() {}

  Hand(const std::vector<Card>& cards);
  Hand(const Card& card);

  Hand& operator+=(const Card& card);
  Hand operator+(const Card& card);

  const unsigned char& size() const { return size_; }
  const int& getSuitHash() const { return suitHash_; }
  const std::array<int, 4>& getSuitBinary() const { return suitBinary_; }
  const std::array<unsigned char, 13> getQuinary() const { return quinary_; }

 private:
  unsigned char size_ = 0;
  int suitHash_ = 0;
  std::array<int, 4> suitBinary_{{0}};
  std::array<unsigned char, 13> quinary_{{0}};
};

} // namespace phevaluator

#endif // __cplusplus

#endif // PHEVALUATOR_HAND_H
