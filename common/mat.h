// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef FAST_MAT_H
#define FAST_MAT_H

#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>


namespace fastnn {

#define ROUND4(x) (((x+3)>>2)<<2)

// the three dimension matrix
class Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4);
    // image
    Mat(int w, int h, size_t elemsize = 4);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    template <typename T>
    void fill(T v);
    // deep copy
    Mat clone() const;
    // reshape vec
    Mat reshape(int w) const;
    // reshape image
    Mat reshape(int w, int h) const;
    // reshape dim
    Mat reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w, size_t elemsize = 4);
    // allocate image
    void create(int w, int h, size_t elemsize = 4);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> T* row(int y);
    template<typename T> const T* row(int y) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // convenient access float vec element
    float& operator[](int i);
    const float& operator[](int i) const;

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // the dimensionality
    int dims;

    int w;
    int h;
    int c;

    int real_c;
};



// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + 2*MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
static inline void NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

inline Mat::Mat()
    : data(0), refcount(0), elemsize(0), dims(0), w(0), h(0), c(0), real_c(0)
{
}

inline Mat::Mat(int _w, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _elemsize);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _elemsize);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _c, _elemsize);
}

inline Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), dims(m.dims)
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    w = m.w;
    h = m.h;
    c = m.c;
    real_c=m.real_c;
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), dims(1)
{
    w = _w;
    h = 1;
    c = 1;
    real_c=1;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), dims(2)
{
    w = _w;
    h = _h;
    c = 1;
    real_c=1;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), dims(3)
{
    w = _w;
    h = _h;
    c = ROUND4(_c);
    real_c=_c;
}

inline Mat::~Mat()
{
    release();
}

inline Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    real_c=m.real_c;

    return *this;
}

inline void Mat::fill(float _v)
{
    int size = total();
    float* ptr = (float*)data;

    int nn = size >> 2;


    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if (nn > 0)
    {
    asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#endif // __aarch64__

}

template <typename T>
inline void Mat::fill(T _v)
{
    int size = total();
    T* ptr = (T*)data;
    for (int i=0; i<size; i++)
    {
        ptr[i] = _v;
    }
}

inline Mat Mat::clone() const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize);
    else if (dims == 2)
        m.create(w, h, elemsize);
    else if (dims == 3)
        m.create(w, h, c, elemsize);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

inline void Mat::create(int _w, size_t _elemsize)
{
    if (dims == 1 && w == _w && elemsize == _elemsize)
        return;

    release();

    elemsize = _elemsize;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    real_c=1;

    if (total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize)
        return;

    release();

    elemsize = _elemsize;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    real_c=1;

    if (total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize)
        return;

    release();

    elemsize = _elemsize;

    dims = 3;
    w = _w;
    h = _h;
    c = ROUND4(_c);

    real_c=_c;

    if (total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
        fastFree(data);

    data = 0;

    elemsize = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    refcount = 0;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Mat::total() const
{
    return w*h*c;
}


inline float* Mat::row(int y)
{
    return (float*)data + w*c* y;
}

inline const float* Mat::row(int y) const
{
    return (const float*)data + w *c* y;
}

template <typename T>
inline T* Mat::row(int y)
{
    return (T*)data + w*c * y;
}

template <typename T>
inline const T* Mat::row(int y) const
{
    return (const T*)data + w*c * y;
}

template <typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}

inline float& Mat::operator[](int i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](int i) const
{
    return ((const float*)data)[i];
}

} // namespace fastnn

#endif // FAST_MAT_H
