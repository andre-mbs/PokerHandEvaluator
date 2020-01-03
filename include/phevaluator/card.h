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

#ifndef PHEVALUATOR_CARD_H
#define PHEVALUATOR_CARD_H

#ifdef __cplusplus

#include <unordered_map>
#include <string>

namespace phevaluator {

class Card {
 public:
  Card() {}

  Card(int id) : id_(id) {}

  Card(std::string name) {
    const std::unordered_map<char, int> rankMap = {
      {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3},
      {'6', 4}, {'7', 5}, {'8', 6}, {'9', 7},
      {'T', 8}, {'J', 9}, {'Q', 10}, {'K', 11}, {'A', 12},
    };
    const std::unordered_map<char, int> suitMap = {
      {'C', 0}, {'D', 1}, {'H', 2}, {'S', 3},
      {'c', 0}, {'d', 1}, {'h', 2}, {'s', 3},
    };

    if (name.length() < 2) {
      // TODO: throw an exception here
    }

    id_ = rankMap.at(name[0]) * 4 + suitMap.at(name[1]);
  }

  Card(const char name[]) : Card(std::string(name)) {}

  operator int() const { return id_; }

 private:
  int id_;
};

} // namespace phevaluator

#endif // __cplusplus

#endif // PHEVALUATOR_CARD_H
