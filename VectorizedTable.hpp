///
/// Copyright 2017 Matt Kaes All Rights Reserved.
///
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#include "immintrin.h" // AVX Intrinsics
#include <cstring> // memset
#include <intrin.h> // _BitScanForward

// Make sure we are able to use AVX2
#ifndef __AVX2__
#ifdef _MSC_VER
#warning "AVX2 is not enabled. Please compile with /arch:AVX2"
#else
#error "AVX2 is not enabled. Please compile with -mavx"
#endif
#endif // !__AVX2__

// This is important to make sure magic AVX alignment works
static_assert(sizeof(int) == 4, "unsigned != 4 bytes! Alignment issues for buckets will occur.");

// Turn Intel warnings into errors
#pragma warning (error: 4752)

#define _m_256i_IntPack(a) _mm256_set_epi32(a,a,a,a,a,a,a,a)
#define AVX_Or(a, b)


// Linear probing Vector Table for unsigned four byte words
// Non-distructing variant
template <typename Value>
class VectorizedTable {
public:
  VectorizedTable()
  {
    m_valueHeap = nullptr;
    m_mm_keyHeap = nullptr;

    grow(MinCapacity);
  }

  ~VectorizedTable()
  {
    delete[] m_valueHeap;
    _mm_free(m_mm_keyHeap);
  }


  Value& add(int index, const Value& value)
  {
    // Catch the issue with repeat adds
    int slot = slotOf(hash(index));
    if (slot != -1)
      return m_valueHeap[slot] = value;

    return insert(hash(index), value);
  }

  bool containsKey(int index) const
  {
    return slotOf(hash(index)) != -1;
  }

  Value& at(int index)
  {
    int slot = slotOf(hash(index));

    if (slot != -1)
      return m_valueHeap[slot];
    else
      return insert(hash(index), Value());
  }

  const Value& operator[](int index) const
  {
    int slot = slotOf(hash(index));

    if (slot != -1)
      return m_valueHeap[slot];
    else
      return Value();
  }

  Value& operator[](int index)
  {
    return at(index);
  }

  void clear()
  {
    m_size = 0;
    std::fill_n(m_mm_keyHeap, m_capacity, EmptySlot);
  }

  int size()
  {
    return m_size;
  }

  int capacity()
  {
    return m_capacity;
  }

private:
  int hash(int x) const {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x & 0x7FFFFFFF;
  }

  Value& insert(int hashIndex, const Value& value)
  {
    if (m_size >= m_capacity / GrowFactor)
      grow(m_capacity * GrowFactor);
    m_size++;

    int slot = openSlot(hashIndex);
    m_mm_keyHeap[slot] = hashIndex;
    return m_valueHeap[slot] = value;
  }

  int openSlot(int hashedIndex) const
  {
    // Create an InUse mask to see if a bucket is full
    const __m256i deadMask = _m_256i_IntPack(DeadSlot);
    int bucketId = ((hashedIndex  % m_capacity) / sizeof(__m256i) * sizeof(int)) / sizeof(int);
    const int bucketCount = m_capacity * sizeof(int) / sizeof(__m256i);
    const int bucketCountBits = bucketCount - 1;

    while (true)
    {
      // life finding
      const __m256i stateMask1 = _mm256_and_si256(byte1Mask, _mm256_cmpgt_epi32(((__m256i*)m_mm_keyHeap)[bucketId], deadMask));
      const __m256i stateMask2 = _mm256_and_si256(byte2Mask, _mm256_cmpgt_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 1], deadMask));
      const __m256i stateMask3 = _mm256_and_si256(byte3Mask, _mm256_cmpgt_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 2], deadMask));
      const __m256i stateMask4 = _mm256_and_si256(byte4Mask, _mm256_cmpgt_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 3], deadMask));
      const int stateMap = ~_mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(_mm256_or_si256(stateMask1, stateMask2), stateMask3), stateMask4));

      unsigned long lowestIndex;
      const long slotFound = _BitScanForward(&lowestIndex, stateMap);

      if (slotFound)
        return (bucketId << 3) + ((lowestIndex & 3) << 3) + (lowestIndex >> 2);

      // Next bucket.
      bucketId = (bucketId + 4) & (bucketCountBits);
    }
  }

  int slotOf(int hashedIndex) const
  {
    // Create an Empty mask to see if we can stop
    const __m256i emptyMask = _m_256i_IntPack(EmptySlot);
    const __m256i indexMask = _m_256i_IntPack(hashedIndex);

    int bucketId = ((hashedIndex  % m_capacity) / sizeof(__m256i) * sizeof(int)) / sizeof(int);
    const int bucketCount = m_capacity * sizeof(int) / sizeof(__m256i);
    const int bucketCountBits = bucketCount - 1;
    int emptyBucket = 0;

    while (true)
    {
      // Value finding
      const __m256i valueMask1 = _mm256_and_si256(byte1Mask, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId], indexMask));
      const __m256i valueMask2 = _mm256_and_si256(byte2Mask, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 1], indexMask));
      const __m256i valueMask3 = _mm256_and_si256(byte3Mask, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 2], indexMask));
      const __m256i valueMask4 = _mm256_and_si256(byte4Mask, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 3], indexMask));

      // Back to scaler mode
      int valueMap = _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(_mm256_or_si256(valueMask1, valueMask2), valueMask3), valueMask4));

      unsigned long lowestIndex;
      const long foundValue = _BitScanForward(&lowestIndex, valueMap);

      if (foundValue)
        return (bucketId << 3) + ((lowestIndex & 3) << 3) + (lowestIndex >> 2);

      // Empty block found
      __m256i empty = _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId], emptyMask);
      empty = _mm256_or_si256(empty, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 1], emptyMask));
      empty = _mm256_or_si256(empty, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 2], emptyMask));
      empty = _mm256_or_si256(empty, _mm256_cmpeq_epi32(((__m256i*)m_mm_keyHeap)[bucketId + 3], emptyMask));

      if (_mm256_movemask_epi8(empty))
        return -1;

      bucketId = (bucketId + 4) & (bucketCountBits);
    }
  }

  void grow(int capacity)
  {
    // Keep a reference to the old pointers
    Value* oldValueHeap = m_valueHeap;
    int* oldKeyHeap = m_mm_keyHeap;
    int oldCapacity = m_capacity;

    // Creating all of the objects
    m_capacity = capacity;
    m_valueHeap = new Value[m_capacity];
    m_mm_keyHeap = (int*)_mm_malloc(m_capacity * sizeof(*m_mm_keyHeap), AVXAlignment);

    clear();

    if (oldKeyHeap)
    {
      for (int i = 0; i < oldCapacity; i++)
        if (oldKeyHeap[i] > DeadSlot)
          insert(oldKeyHeap[i], oldValueHeap[i]);

      delete[] oldValueHeap;
      _mm_free(oldKeyHeap);
    }
  }

  const int EmptySlot = (int)0x80000000;
  const int DeadSlot = (int)0x80000001;

  Value* m_valueHeap;
  int* m_mm_keyHeap;

  int m_size;
  int m_capacity;

  const int AVXAlignment = 16;
  const int MinCapacity = 64;
  const int GrowFactor = 4;

  const __m256i byte1Mask = _m_256i_IntPack(0x000000FF);
  const __m256i byte2Mask = _m_256i_IntPack(0x0000FF00);
  const __m256i byte3Mask = _m_256i_IntPack(0x00FF0000);
  const __m256i byte4Mask = _m_256i_IntPack(0xFF000000);
};