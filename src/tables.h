/*
 *  Copyright 2016 Henry Lee
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

#ifndef TABLES_H
#define TABLES_H

extern const short flush[8192];
extern const short noflush5[6175];
extern const short noflush6[18395];
extern const short noflush7[49205];
extern const short noflush8[120055];
extern const short noflush9[270270];
extern const unsigned char suits[4609];

extern const unsigned int choose[53][10];
extern const int dp[5][14][10];

extern const char* rank_description[7463][2];

extern const short noflush_omaha[11238500];
extern const short flush_omaha[4099095];

#endif // TABLES_H
