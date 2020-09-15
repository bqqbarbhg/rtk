#include "rtk.h"

// -- Temp options

#define RTK_BVH_MAX_DEPTH 32

#define RTKI_MAX_CONCURRENT_TASKS 256

// #define RTK_USE_DOUBLE 1

#include <math.h>
#include <assert.h>

#define rtk_assert(x) do { if (!(x)) __debugbreak(); } while (0)

// -- Compiler builtins

#ifndef RTK_USE_DOUBLE
#define RTK_USE_DOUBLE 0
#endif

#if defined(_MSC_VER)

#include <intrin.h>
#define rtki_inline static __forceinline

#include <intrin.h>
#define rtki_firstbit4(dst_index, mask) _BitScanForward((unsigned long*)&(dst_index), (unsigned long)(mask))
#define rtki_popcount4(mask) (uint32_t)_mm_popcnt_u32((unsigned int)(mask))
#define RTKI_ALIGN16 __declspec(align(16))

#if defined(_M_X64)
	typedef uint64_t rtki_atomic_size;
	#define rtki_atomic_size_add(a, v) (size_t)_InterlockedExchangeAdd64((volatile long long*)(a), (long long)(v))
#elif defined(_M_IX86)
	typedef uint32_t rtki_atomic_size;
	#define rtki_atomic_size_add(a, v) (size_t)_InterlockedExchangeAdd((volatile long*)(a), (long)(v))
#endif

#if !RTK_USE_DOUBLE && !defined(RTK_USE_SSE) && (defined(_M_X64) || defined(_M_IX86))
	#define RTK_USE_SSE 1
	#include <xmmintrin.h>
	#include <emmintrin.h>
	#include <emmintrin.h>
#endif

#ifndef rtk_mem_alloc
	#include <malloc.h>
	#define rtk_mem_alloc(size) _aligned_malloc(size, 64)
	#define rtk_mem_free(ptr, size) _aligned_free(ptr)
#endif

#elif defined(__GNUC__)

#define rtki_inline static inline __attribute__((always_inline))
#error "TODO"

#else
#error "TODO: C++ fallback"
#endif

#ifndef RTK_USE_SSE
#define RTK_USE_SSE 0
#endif

// -- Utility

rtki_inline size_t rtki_min_sz(size_t a, size_t b) { return a < b ? a : b; }
rtki_inline size_t rtki_max_sz(size_t a, size_t b) { return a < b ? b : a; }
rtki_inline size_t rtki_align_up_sz(size_t v, size_t align) { return v + ((size_t)-(intptr_t)v & (align - 1)); }


// -- Allocation

#define RTKI_MEM_MAGIC 0x494b5452 // 'RTKI'
typedef struct rtki_mem_footer {
	uint32_t magic;
	size_t alloc_size;
} rtki_mem_footer;

void *rtki_mem_alloc(size_t size)
{
	size_t alloc_size = size + sizeof(rtki_mem_footer);
	void *ptr = rtk_mem_alloc(alloc_size);
	if (!ptr) return NULL;
	rtki_mem_footer *footer = (rtki_mem_footer*)((char*)ptr + size);
	footer->magic = RTKI_MEM_MAGIC;
	footer->alloc_size = size;
	return ptr;
}

void rtki_mem_free(void *ptr, size_t size)
{
	rtk_assert(ptr);
	rtki_mem_footer *footer = (rtki_mem_footer*)((char*)ptr + size);

	// RTK allocation footer mismatch, either you're trying to free a pointer
	// not allocated by RTK or something corrupted memory after the allocation.
	rtk_assert(footer->magic == RTKI_MEM_MAGIC);
	rtk_assert(footer->alloc_size == size);

	rtk_mem_free(ptr, size);
}

// -- In-register vec3 + some intrinsics

#if RTK_USE_SSE

typedef __m128 rtki_reg3;

#define rtki_real_min(a, b) _mm_cvtss_f32(_mm_min_ss(_mm_set_ss((a)), _mm_set_ss((b))))
#define rtki_real_max(a, b) _mm_cvtss_f32(_mm_max_ss(_mm_set_ss((a)), _mm_set_ss((b))))

rtki_inline rtki_reg3 rtki_load3(const rtk_vec3 *v) { return _mm_setr_ps(v->x, v->y, v->z, v->z); }
rtki_inline rtki_reg3 rtki_load3x(const rtk_vec3 *v) {
	__m128 t = _mm_loadu_ps(&v->x);
	return _mm_shuffle_ps(t, t, _MM_SHUFFLE(2,2,1,0));
}
rtki_inline rtki_reg3 rtki_loadx3(const rtk_vec3 *v) {
	__m128 t = _mm_loadu_ps(&v->x - 1);
	return _mm_shuffle_ps(t, t, _MM_SHUFFLE(3,3,2,1));
}
#define rtki_broadcast3(x) _mm_set1_ps((x))
#define rtki_add3(a, b) _mm_add_ps((a), (b))
#define rtki_sub3(a, b) _mm_sub_ps((a), (b))
#define rtki_mul3(a, b) _mm_mul_ps((a), _mm_set_ss(b))
#define rtki_mulcomp3(a, b) _mm_mul_ps((a), (b))
#define rtki_min3(a, b) _mm_min_ps((a), (b))
#define rtki_max3(a, b) _mm_max_ps((a), (b))
#define rtki_abs3(a) _mm_andnot_ps(_mm_set1_ps(-0.0f), (a))
#define rtki_x3(a) _mm_cvtss_f32((a))
rtki_inline float rtki_y3(rtki_reg3 r) { return _mm_cvtss_f32(_mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,1))); }
rtki_inline float rtki_z3(rtki_reg3 r) { return _mm_cvtss_f32(_mm_movehl_ps(r, r)); }
rtki_inline rtk_vec3 rtki_to_vec3(rtki_reg3 r) {
	rtk_vec3 v = { rtki_x3(r), rtki_y3(r), rtki_z3(r) };
	return v;
}
rtki_inline rtk_real rtki_maxcomp3(rtki_reg3 r) {
	__m128 tmp = _mm_max_ps(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,1)));
	return _mm_cvtss_f32(_mm_max_ps(tmp, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,2))));
}
rtki_inline rtk_real rtki_mincomp3(rtki_reg3 r) {
	__m128 tmp = _mm_min_ps(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,1)));
	return _mm_cvtss_f32(_mm_min_ps(tmp, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,2))));
}
rtki_inline uint32_t rtki_signmask3(rtki_reg3 r) {
	return (uint32_t)_mm_movemask_ps(r) & 0x7u;
}

#else

#if defined(__GNUC__)
	#if RTK_USE_DOUBLE
		#define rtki_real_abs(a) __builtin_fabs(a)
	#else
		#define rtki_real_abs(a) __builtin_fabsf(a)
	#endif
#else
	#if RTK_USE_DOUBLE
		#define rtki_real_abs(a) fabs(a)
	#else
		#define rtki_real_abs(a) fabsf(a)
	#endif
#endif

rtk_real rtki_real_min(rtk_real a, rtk_real b) { return a < b ? a : b; }
rtk_real rtki_real_max(rtk_real a, rtk_real b) { return a < b ? b : a; }

typedef rtk_vec3 rtki_reg3;

#define rtki_load3(v) (*(v))
#define rtki_load3x(v) (*(v))
#define rtki_loadx3(v) (*(v))
rtki_inline rtki_reg3 rtki_broadcast3(rtk_real x) { rtki_reg3 v = { x, x, x }; return v; }
rtki_inline rtki_reg3 rtki_add3(rtki_reg3 a, rtki_reg3 b) { rtki_reg3 v = { a.x+b.x, a.y+b.y, a.z+b.z }; return v; }
rtki_inline rtki_reg3 rtki_sub3(rtki_reg3 a, rtki_reg3 b) { rtki_reg3 v = { a.x-b.x, a.y-b.y, a.z-b.z }; return v; }
rtki_inline rtki_reg3 rtki_mul3(rtki_reg3 a, rtk_real b) { rtki_reg3 v = { a.x-b, a.y-b, a.z-b }; return v; }
rtki_inline rtki_reg3 rtki_mulcomp3(rtki_reg3 a, rtk_reg3 b) { rtki_reg3 v = { a.x-b.x, a.y-b.y, a.z-b.z }; return v; }
rtki_inline rtki_reg3 rtki_min3(rtki_reg3 a, rtki_reg3 b) {
	rtki_reg3 v = { rtki_real_min(a.x, b.x), rtki_real_min(a.y, b.y), rtki_real_min(a.z, b.z) };
	return v;
}
rtki_inline rtki_reg3 rtki_max3(rtki_reg3 a, rtki_reg3 b) {
	rtki_reg3 v = { rtki_real_max(a.x, b.x), rtki_real_max(a.y, b.y), rtki_real_max(a.z, b.z) };
	return v;
}
rtki_inline rtki_reg3 rtki_abs3(rtki_reg3 a, rtki_reg3 b) {
	rtki_reg3 v = { rtki_real_abs(a.x, b.x), rtki_real_abs(a.y, b.y), rtki_real_abs(a.z, b.z) };
	return v;
}
#define rtki_x3(a) ((a).x)
#define rtki_y3(a) ((a).y)
#define rtki_z3(a) ((a).z)
#define rtki_to_vec3(a) (a)
rtki_inline rtk_real rtki_maxcomp3(rtki_reg3 r) {
	return rtki_real_max(rtki_real_max(r.x, r.y), r.z);
}
rtki_inline rtk_real rtki_mincomp3(rtki_reg3 r) {
	return rtki_real_min(rtki_real_min(r.x, r.y), r.z);
}
rtki_inline uint32_t rtki_signmask3(rtki_reg3 r) {
	uint32_t mask = 0;
	mask |= signbit(r.x) ? 0x1 : 0;
	mask |= signbit(r.y) ? 0x2 : 0;
	mask |= signbit(r.z) ? 0x4 : 0;
	return mask;
}

#endif

typedef struct {
	rtki_reg3 min, max;
} rtki_regbounds;

rtki_inline void rtki_regbounds_reset(rtki_regbounds *bounds) {
	bounds->min = rtki_broadcast3(+RTK_INF);
	bounds->max = rtki_broadcast3(-RTK_INF);
}

rtki_inline void rtki_bounds_add(rtki_regbounds *dst, const rtki_regbounds *a) {
	dst->min = rtki_min3(dst->min, a->min);
	dst->max = rtki_max3(dst->max, a->max);
}

rtki_inline void rtki_bounds_add_point(rtki_regbounds *dst, rtki_reg3 a) {
	dst->min = rtki_min3(dst->min, a);
	dst->max = rtki_max3(dst->max, a);
}

typedef struct rtki_trace {
	const char *data;

	rtk_ray ray;
	rtk_hit hit;

	rtk_vec3 rcp_ray_dir;
	rtk_vec3 shear_origin;
	rtk_vec3 shear;

	uint32_t sign_mask;
	uint32_t shear_x;
	uint32_t shear_y;
	uint32_t shear_z;
} rtki_trace;

#if RTK_USE_SSE

// -- SSE Implementation

#define rtki_mm_broadcast(reg, lane) _mm_shuffle_ps((reg), (reg), _MM_SHUFFLE(lane, lane, lane, lane))
#define rtki_mm_min3(a, b, c) _mm_min_ps(_mm_min_ps((a), (b)), (c))
#define rtki_mm_max3(a, b, c) _mm_max_ps(_mm_max_ps((a), (b)), (c))
#define rtki_mm_min4(a, b, c, d) _mm_min_ps(_mm_min_ps((a), (b)), _mm_min_ps((c), (d)))
#define rtki_mm_max4(a, b, c, d) _mm_max_ps(_mm_max_ps((a), (b)), _mm_max_ps((c), (d)))
#define rtki_mm_submul(a, b, c) _mm_mul_ps(_mm_sub_ps(rtki_mm_load_f32((a)), (b)), (c))
#define rtki_mm_get_f32(a, ix) _mm_cvtss_f32(_mm_shuffle_ps((a), (a), _MM_SHUFFLE(3,2,1,ix)))
#define rtki_mm_get_u32(a, ix) ((uint32_t)_mm_cvtsi128_si32(_mm_shuffle_epi32((a), _MM_SHUFFLE(3,2,1,ix))))
#define rtki_mm_rcp(a) _mm_div_ps(_mm_set1_ps(1.0f), (a))

#if 1
	#define rtki_mm_load_f32(ptr) _mm_loadu_ps((const float*)(ptr))
	#define rtki_mm_load_si128(ptr) _mm_loadu_si128((const __m128i*)(ptr))
#else
	#define rtki_mm_load_f32(ptr) _mm_load_ps((const float*)(ptr))
	#define rtki_mm_load_si128(ptr) _mm_load_si128((const __m128i*)(ptr))
#endif
#if 1
	#define rtki_mm_blend(a, b, mask) _mm_blendv_ps((a), (b), (mask))
#else
	#define rtki_mm_blend(a, b, mask) _mm_or_ps(_mm_andnot_ps((mask), (b)), _mm_and_ps((mask), (a)));
#endif

static const uint32_t shuf_u32_tab[4] = {
	0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
};

static void rtki_leaf_traverse_sse(rtki_trace *rt, rtk_offset leaf_offset)
{
	const char *const data = rt->data;
	rtk_assert((leaf_offset & 1) == 1);

	const rtk_leaf *const leaf = (const rtk_leaf*)(data + (leaf_offset^1));
	const size_t num_tris = leaf->num_triangles;
	const size_t num_prims = leaf->num_primitives;
	const size_t num_tris_aligned = (num_tris + 3) & ~3u;

	const rtk_vertex *verts = (const rtk_vertex*)(data + leaf->vertex_data);
	const rtk_leaf_triangle *tris = (const rtk_leaf_triangle*)(leaf + 1);
	const rtk_leaf_primitive *prims = (const rtk_leaf_primitive*)(tris + num_tris_aligned);
	const rtk_leaf_mesh *meshes = (const rtk_leaf_mesh*)(prims + num_prims);

	const __m128 shear_origin_x = _mm_set1_ps(rt->shear_origin.x);
	const __m128 shear_origin_y = _mm_set1_ps(rt->shear_origin.y);
	const __m128 shear_origin_z = _mm_set1_ps(rt->shear_origin.z);

	const __m128 shear_x = _mm_set1_ps(rt->shear.x);
	const __m128 shear_y = _mm_set1_ps(rt->shear.y);
	const __m128 shear_z = _mm_set1_ps(rt->shear.z);

	// Setup the _mm_shuffle_epi8() tables
	const __m128i shear_shuf = _mm_setr_epi32((int)shuf_u32_tab[rt->shear_x],
		(int)shuf_u32_tab[rt->shear_y], (int)shuf_u32_tab[rt->shear_z], -1);

	// Ray distance limits
	const __m128 min_t = _mm_set1_ps(rt->ray.min_t);
	__m128 max_t = _mm_set1_ps(rt->hit.t);

	const rtk_leaf_triangle *tris_end = tris + num_tris_aligned;
	for (; tris != tris_end; tris += 4) {

		// Load AOS triangle data (XYZ + vertex index)
		const __m128i ai0 = rtki_mm_load_si128(verts + tris[0].leaf_vertices[0]);
		const __m128i bi0 = rtki_mm_load_si128(verts + tris[1].leaf_vertices[0]);
		const __m128i ci0 = rtki_mm_load_si128(verts + tris[2].leaf_vertices[0]);
		const __m128i di0 = rtki_mm_load_si128(verts + tris[3].leaf_vertices[0]);

		const __m128i ai1 = rtki_mm_load_si128(verts + tris[0].leaf_vertices[1]);
		const __m128i bi1 = rtki_mm_load_si128(verts + tris[1].leaf_vertices[1]);
		const __m128i ci1 = rtki_mm_load_si128(verts + tris[2].leaf_vertices[1]);
		const __m128i di1 = rtki_mm_load_si128(verts + tris[3].leaf_vertices[1]);

		const __m128i ai2 = rtki_mm_load_si128(verts + tris[0].leaf_vertices[2]);
		const __m128i bi2 = rtki_mm_load_si128(verts + tris[1].leaf_vertices[2]);
		const __m128i ci2 = rtki_mm_load_si128(verts + tris[2].leaf_vertices[2]);
		const __m128i di2 = rtki_mm_load_si128(verts + tris[3].leaf_vertices[2]);

		// Permute the coordinate axes to shear space

		const __m128 a0 = _mm_castsi128_ps(_mm_shuffle_epi8(ai0, shear_shuf));
		const __m128 b0 = _mm_castsi128_ps(_mm_shuffle_epi8(bi0, shear_shuf));
		const __m128 c0 = _mm_castsi128_ps(_mm_shuffle_epi8(ci0, shear_shuf));
		const __m128 d0 = _mm_castsi128_ps(_mm_shuffle_epi8(di0, shear_shuf));
		const __m128 a1 = _mm_castsi128_ps(_mm_shuffle_epi8(ai1, shear_shuf));
		const __m128 b1 = _mm_castsi128_ps(_mm_shuffle_epi8(bi1, shear_shuf));
		const __m128 c1 = _mm_castsi128_ps(_mm_shuffle_epi8(ci1, shear_shuf));
		const __m128 d1 = _mm_castsi128_ps(_mm_shuffle_epi8(di1, shear_shuf));
		const __m128 a2 = _mm_castsi128_ps(_mm_shuffle_epi8(ai2, shear_shuf));
		const __m128 b2 = _mm_castsi128_ps(_mm_shuffle_epi8(bi2, shear_shuf));
		const __m128 c2 = _mm_castsi128_ps(_mm_shuffle_epi8(ci2, shear_shuf));
		const __m128 d2 = _mm_castsi128_ps(_mm_shuffle_epi8(di2, shear_shuf));

		// Transpose the triangle coordinates/indices to SOA

		__m128 t0, t1, vx, vy, vz;

		t0 = _mm_unpacklo_ps(a0, b0); // XX YY
		t1 = _mm_unpacklo_ps(c0, d0); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a0, b0); // ZZ II
		t1 = _mm_unpackhi_ps(c0, d0); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v0x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v0y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v0z = _mm_sub_ps(vz, shear_origin_z);

		t0 = _mm_unpacklo_ps(a1, b1); // XX YY
		t1 = _mm_unpacklo_ps(c1, d1); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a1, b1); // ZZ II
		t1 = _mm_unpackhi_ps(c1, d1); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v1x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v1y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v1z = _mm_sub_ps(vz, shear_origin_z);

		t0 = _mm_unpacklo_ps(a2, b2); // XX YY
		t1 = _mm_unpacklo_ps(c2, d2); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a2, b2); // ZZ II
		t1 = _mm_unpackhi_ps(c2, d2); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v2x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v2y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v2z = _mm_sub_ps(vz, shear_origin_z);

		// Transform the triangle coordinates to sheared ray space

		const __m128 x0 = _mm_add_ps(v0x, _mm_mul_ps(shear_x, v0z));
		const __m128 y0 = _mm_add_ps(v0y, _mm_mul_ps(shear_y, v0z));
		const __m128 z0 = _mm_mul_ps(shear_z, v0z);
		const __m128 x1 = _mm_add_ps(v1x, _mm_mul_ps(shear_x, v1z));
		const __m128 y1 = _mm_add_ps(v1y, _mm_mul_ps(shear_y, v1z));
		const __m128 z1 = _mm_mul_ps(shear_z, v1z);
		const __m128 x2 = _mm_add_ps(v2x, _mm_mul_ps(shear_x, v2z));
		const __m128 y2 = _mm_add_ps(v2y, _mm_mul_ps(shear_y, v2z));
		const __m128 z2 = _mm_mul_ps(shear_z, v2z);

		// Calculate the edge functions u,v,w. If any of these functions is
		// exactly zero recalculate them all using double precision to
		// guarantee watertight intersections.

		__m128 u = _mm_sub_ps(_mm_mul_ps(x1, y2), _mm_mul_ps(y1, x2));
		__m128 v = _mm_sub_ps(_mm_mul_ps(x2, y0), _mm_mul_ps(y2, x0));
		__m128 w = _mm_sub_ps(_mm_mul_ps(x0, y1), _mm_mul_ps(y0, x1));
		const __m128 zero = _mm_setzero_ps();
		const __m128 zero_u = _mm_cmpeq_ps(u, zero);
		const __m128 zero_v = _mm_cmpeq_ps(v, zero);
		const __m128 zero_w = _mm_cmpeq_ps(w, zero);
		const __m128 any_zero = _mm_or_ps(_mm_or_ps(zero_u, zero_v), zero_w);
		if (_mm_movemask_ps(any_zero) != 0) {
			__m128d xd0 = _mm_cvtps_pd(x0);
			__m128d yd0 = _mm_cvtps_pd(y0);
			__m128d xd1 = _mm_cvtps_pd(x1);
			__m128d yd1 = _mm_cvtps_pd(y1);
			__m128d xd2 = _mm_cvtps_pd(x2);
			__m128d yd2 = _mm_cvtps_pd(y2);

			__m128d ud = _mm_sub_pd(_mm_mul_pd(xd1, yd2), _mm_mul_pd(yd1, xd2));
			__m128d vd = _mm_sub_pd(_mm_mul_pd(xd2, yd0), _mm_mul_pd(yd2, xd0));
			__m128d wd = _mm_sub_pd(_mm_mul_pd(xd0, yd1), _mm_mul_pd(yd0, xd1));

			u = _mm_cvtpd_ps(ud);
			v = _mm_cvtpd_ps(vd);
			w = _mm_cvtpd_ps(wd);

			xd0 = _mm_cvtps_pd(_mm_movehl_ps(x0, x0));
			yd0 = _mm_cvtps_pd(_mm_movehl_ps(y0, y0));
			xd1 = _mm_cvtps_pd(_mm_movehl_ps(x1, x1));
			yd1 = _mm_cvtps_pd(_mm_movehl_ps(y1, y1));
			xd2 = _mm_cvtps_pd(_mm_movehl_ps(x2, x2));
			yd2 = _mm_cvtps_pd(_mm_movehl_ps(y2, y2));

			ud = _mm_sub_pd(_mm_mul_pd(xd1, yd2), _mm_mul_pd(yd1, xd2));
			vd = _mm_sub_pd(_mm_mul_pd(xd2, yd0), _mm_mul_pd(yd2, xd0));
			wd = _mm_sub_pd(_mm_mul_pd(xd0, yd1), _mm_mul_pd(yd0, xd1));

			u = _mm_movelh_ps(u, _mm_cvtpd_ps(ud));
			v = _mm_movelh_ps(v, _mm_cvtpd_ps(vd));
			w = _mm_movelh_ps(w, _mm_cvtpd_ps(wd));
		}

		// Only if all u,v,w are either negative or positive we hit the triangle,
		// so we can bail out if both signs are represented
		const __m128 neg = _mm_cmplt_ps(_mm_min_ps(_mm_min_ps(u, v), w), zero);
		const __m128 pos = _mm_cmpgt_ps(_mm_max_ps(_mm_max_ps(u, v), w), zero);
		const __m128 bad_sign = _mm_and_ps(neg, pos);
		int bad_sign_mask = _mm_movemask_ps(bad_sign);
		if (bad_sign_mask == 0xf) continue;

		__m128 det = _mm_add_ps(_mm_add_ps(u, v), w);
		__m128 rcp_det = rtki_mm_rcp(det);

		__m128 z = _mm_mul_ps(u, z0);
		z = _mm_add_ps(z, _mm_mul_ps(v, z1));
		z = _mm_add_ps(z, _mm_mul_ps(w, z2));

		__m128 t = _mm_mul_ps(z, rcp_det);
		__m128 good = _mm_and_ps(_mm_cmpgt_ps(t, min_t), _mm_cmplt_ps(t, max_t));

		// Scalar loop over hit triangles
		int good_mask = _mm_movemask_ps(good);
		good_mask &= ~bad_sign_mask;
		if (good_mask != 0) {
			RTKI_ALIGN16 float ts[4], us[4], vs[4];

			_mm_store_ps(ts, t);
			_mm_store_ps(us, _mm_mul_ps(u, rcp_det));
			_mm_store_ps(vs, _mm_mul_ps(v, rcp_det));

			do {
				uint32_t lane_ix;
				rtki_firstbit4(lane_ix, good_mask);

				float lane_t = ts[lane_ix];
				if (lane_t < rt->hit.t) {
					const rtk_leaf_triangle *tri = &tris[lane_ix];
					const rtk_leaf_mesh *mesh = &meshes[tri->leaf_mesh];
					rt->hit.t = lane_t;
					rt->hit.u = us[lane_ix];
					rt->hit.v = vs[lane_ix];
					rt->hit.vertex[0] = verts[tri->leaf_vertices[0]];
					rt->hit.vertex[1] = verts[tri->leaf_vertices[1]];
					rt->hit.vertex[2] = verts[tri->leaf_vertices[2]];
					rt->hit.mesh_index = mesh->mesh_index;
					rt->hit.triangle_index = tri->triangle_index;
					rt->hit.user_data = mesh->user_data;
					max_t = _mm_set1_ps(lane_t);
				}

				good_mask &= good_mask - 1;
			} while (good_mask != 0);
		}
	}

	const rtki_reg3 origin = rtki_load3x(&rt->ray.origin);
	const rtki_reg3 rcp_dir = rtki_loadx3(&rt->rcp_ray_dir);
	const rtk_leaf_primitive *prims_end = prims + num_prims;
	for (; prims != prims_end; prims++) {

		const rtki_reg3 min = _mm_loadu_ps(prims->bounds.min.v);
		const rtki_reg3 max = _mm_loadu_ps(prims->bounds.max.v);
		const rtki_reg3 t_lo = rtki_mulcomp3(rtki_sub3(min, origin), rcp_dir);
		const rtki_reg3 t_hi = rtki_mulcomp3(rtki_sub3(max, origin), rcp_dir);
		const rtki_reg3 t_min = rtki_min3(t_lo, t_hi);
		const rtki_reg3 t_max = rtki_max3(t_lo, t_hi);
		rtk_real min_t = rtki_maxcomp3(t_min);
		rtk_real max_t = rtki_mincomp3(t_max);
		if (min_t <= max_t && max_t > rt->ray.min_t && max_t < rt->hit.t) {
			rt->hit.t = min_t;
			rt->hit.mesh_index = ~0u;
			rt->hit.triangle_index = ~0u;
			rt->hit.primitive_index = prims->prim_index;
			rt->hit.user_data = prims->user_data;
		}
	}

}

static void rtki_bvh_traverse_sse(rtki_trace *rt, rtk_offset root_offset)
{
	const char *const data = rt->data;
	rtk_assert((root_offset & 1) == 0);

	// Traversal stack: The top node is kept separate in registers,
	// the stack only contains the pushed t/ptr values. Initialize
	// the stack so that it's safe to pop once without checking 
	// that the stack is exhausted.
	float top_t = -RTK_INF;
	rtk_offset top_ptr = root_offset;
	float stack_t[RTK_BVH_MAX_DEPTH + 1];
	uint64_t stack_ptr[RTK_BVH_MAX_DEPTH + 1];
	uint32_t stack_depth = 2;
	stack_t[1] = stack_t[0] = RTK_INF;
	stack_ptr[1] = stack_ptr[0] = 0;

	const __m128 pos_inf = _mm_set1_ps(3.402823e+38f);

	const __m128 ray_origin = _mm_loadu_ps(&rt->ray.origin.x);
	const __m128 ray_rcp_dir = _mm_loadu_ps(&rt->rcp_ray_dir.x);
	const __m128 origin_x = rtki_mm_broadcast(ray_origin, 0);
	const __m128 origin_y = rtki_mm_broadcast(ray_origin, 1);
	const __m128 origin_z = rtki_mm_broadcast(ray_origin, 2);
	const __m128 rcp_dir_x = rtki_mm_broadcast(ray_rcp_dir, 0);
	const __m128 rcp_dir_y = rtki_mm_broadcast(ray_rcp_dir, 1);
	const __m128 rcp_dir_z = rtki_mm_broadcast(ray_rcp_dir, 2);
	const __m128 ray_min_t = _mm_load1_ps(&rt->ray.min_t);

	const uint32_t sign_mask = rt->sign_mask;
	const uint32_t sign_x = sign_mask & 1;
	const uint32_t sign_y = sign_mask >> 1 & 1;
	const uint32_t sign_z = sign_mask >> 2;

	const __m128 tag_mask = _mm_castsi128_ps(_mm_set1_epi32(0x3));
	const __m128 tag_indices = _mm_castsi128_ps(_mm_setr_epi32(0, 1, 2, 3));

	for (;;) {
		const float ray_hit_t = rt->hit.t;

		// Pop nodes while our current geometry hit distance is less
		// than the current top BVH hit distance.
		while (top_t >= ray_hit_t) {
			rtk_assert(stack_depth > 0);
			if (--stack_depth == 0) return;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
		}

		// Now we have a valid top BVH node! If it's a leaf traverse the
		// primitives and pop the stack and `continue` to return to it.
		if ((top_ptr & 1) != 0) {
			rtki_leaf_traverse_sse(rt, top_ptr);
			--stack_depth;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
			continue;
		}

		// The top BVH node is an internal node: Intersect the ray against the
		// axis+aligned planes and find the intersection distances `min_t` and `max_t`.
		// We want to traverse to the child BVH node if all:
		// (1) `min_t <= max_t`: The ray intersects the AABB
		// (2) `min_t < ray_hit_t`: The AABB intersection distance is closer than ray hit/max
		// (3) `max_t > ray_min_t`: The far AABB distance is further than the ray min
		// We can fold all these tests into a single expression:
		// `max(min_t, ray_min_t) <= min(max_t, ray_hit_t)`
		const rtk_bvh_node *node = (const rtk_bvh_node*)(data + (size_t)top_ptr);
		const __m128 min_x = rtki_mm_submul(node->bounds_x[sign_x    ], origin_x, rcp_dir_x);
		const __m128 max_x = rtki_mm_submul(node->bounds_x[sign_x ^ 1], origin_x, rcp_dir_x);
		const __m128 min_y = rtki_mm_submul(node->bounds_y[sign_y    ], origin_y, rcp_dir_y);
		const __m128 max_y = rtki_mm_submul(node->bounds_y[sign_y ^ 1], origin_y, rcp_dir_y);
		const __m128 min_z = rtki_mm_submul(node->bounds_z[sign_z    ], origin_z, rcp_dir_z);
		const __m128 max_z = rtki_mm_submul(node->bounds_z[sign_z ^ 1], origin_z, rcp_dir_z);
		const __m128 min_t = rtki_mm_max4(min_x, min_y, min_z, ray_min_t);
		const __m128 max_t = rtki_mm_min4(max_x, max_y, max_z, _mm_set1_ps(ray_hit_t));
		const __m128 mask = _mm_and_ps(_mm_cmple_ps(min_t, max_t),  // (1)
			_mm_and_ps(_mm_cmplt_ps(min_t, _mm_set1_ps(ray_hit_t)), // (2)
				_mm_cmplt_ps(ray_min_t, max_t)));                   // (3)

		// Make a mask of nodes to traverse, replace t values for inactive lanes with
		// infinity and count the number of hit node bits.
		const __m128 ts = rtki_mm_blend(pos_inf, min_t, mask);
		uint32_t mask_bits = (uint32_t)_mm_movemask_ps(mask);
		uint32_t num_bits = rtki_popcount4(mask_bits);

		if (mask_bits == 0) {
			// Hit nothing: Pop the BVH node stack
			rtk_assert(stack_depth > 0);
			--stack_depth;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
		} else if (num_bits == 1) {
			// Hit a single node: Replace the top node with the minimum distance
			// and first (only) found bit index.
			uint32_t lowest_bit;
			rtki_firstbit4(lowest_bit, mask_bits);
			const __m128 tmp = _mm_min_ps(ts, _mm_shuffle_ps(ts, ts, _MM_SHUFFLE(2,3,0,1))); 
			top_t = _mm_cvtss_f32(_mm_min_ps(tmp, _mm_movehl_ps(tmp, tmp)));
			top_ptr = node->tagged_offsets[lowest_bit];
		} else {
			// Hit 2-4 nodes: Tag the hit distances with indices (two lowest bits)
			// and sort them inside the XMM registers using a sorting network:
			// t_tag[0] --v--- t_lo[01] - t_1[0] --v--- t_lo[01] - t_2[0] ----- t_lo/hi[0] - t_tag_sorted[0]
			// t_tag[1] --^--- t_hi[01] - t_1[2] --|v-- t_lo[23] - t_2[1] --v--  t_lo[12]  - t_tag_sorted[1]
			// t_tag[2] ---v-- t_lo[23] - t_1[1] --^|-- t_hi[01] - t_2[2] --^--  t_hi[12]  - t_tag_sorted[2]
			// t_tag[3] ---^-- t_hi[23] - t_1[3] ---^-- t_hi[23] - t_2[3] ----- t_lo/hi[3] - t_tag_sorted[3]
			const __m128 t_tag = _mm_or_ps(_mm_andnot_ps(tag_mask, ts), tag_indices);

			__m128 t_swap = _mm_shuffle_ps(t_tag, t_tag, _MM_SHUFFLE(2,3,0,1));
			__m128 t_lo = _mm_min_ps(t_tag, t_swap);
			__m128 t_hi = _mm_max_ps(t_tag, t_swap);
			const __m128 t_1 = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(2,0,2,0));

			t_swap = _mm_shuffle_ps(t_1, t_1, _MM_SHUFFLE(2,3,0,1));
			t_lo = _mm_min_ps(t_1, t_swap);
			t_hi = _mm_max_ps(t_1, t_swap);
			const __m128 t_2 = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(2,0,2,0));

			t_swap = _mm_shuffle_ps(t_2, t_2, _MM_SHUFFLE(3,1,2,0));
			t_lo = _mm_min_ps(t_2, t_swap);
			t_hi = _mm_max_ps(t_2, t_swap);

			const __m128 t_tag_sorted = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(3,2,1,0));

			// Clear the tag bits from the sorted values, this means we might
			// unnecessarily enter nodes that are 4 bits away from the max distance.
			const __m128 t_sorted = _mm_andnot_ps(tag_mask, t_tag_sorted);
			const __m128i tag_sorted = _mm_castps_si128(_mm_and_ps(tag_mask, t_tag_sorted));

			// Traverse the top hit directly and push the rest to the stack
			stack_depth += num_bits - 1;
			float *dst_t = stack_t + stack_depth;

			rtk_offset *dst_ptr = stack_ptr + stack_depth;
			top_t = rtki_mm_get_f32(t_sorted, 0);
			top_ptr = node->tagged_offsets[rtki_mm_get_u32(tag_sorted, 0)];

			dst_t[0] = rtki_mm_get_f32(t_sorted, 1);
			dst_ptr[0] = node->tagged_offsets[rtki_mm_get_u32(tag_sorted, 1)];
			if (num_bits >= 3) {
				dst_t[1] = rtki_mm_get_f32(t_sorted, 2);
				dst_ptr[1] = node->tagged_offsets[rtki_mm_get_u32(tag_sorted, 2)];
				if (num_bits == 4) {
					dst_t[2] = rtki_mm_get_f32(t_sorted, 3);
					dst_ptr[2] = node->tagged_offsets[rtki_mm_get_u32(tag_sorted, 3)];
				}
			}
		}
	}
}

#endif

// -- Scene building

typedef struct {
	size_t in_base, in_num; // < Input: Range of triangles to process
	rtk_bounds out_bounds;  // < Output: Bounding box
} rtki_setup_triangle_state;

typedef struct {
	size_t in_base, in_num; // < Input: Range of triangles to process
	rtk_bounds out_bounds;  // < Output: Bounding box
} rtki_setup_primitive_state;

typedef struct {
	rtk_bounds bounds;        // < AABB of the item
	rtk_vertex vertex[3];     // < Triangle vertices
	uint32_t mesh_prim_index; // < Mesh or primitive index, check `triangle_index` to see which one
	uint32_t triangle_index;  // < Triangle in mesh, ~0u if primitive
	uint8_t v_ix[3];          // < Local de-duplicated vertex indices
} rtki_build_item;

typedef struct {
	rtk_bounds bounds;    // < AABB of the BVH node
	size_t item_begin;    // < Start of the BVH item range
	size_t item_count;    // < Number of BVH items in the node
	size_t child_index;   // < Index of children or SIZE_MAX if leaf
	size_t vertex_offset; // < Closed vertex data offset or SIZE_MAX if not closed
	uint16_t depth;       // < Depth in the tree
} rtki_build_node;

typedef struct {
	size_t alloc_size;
	size_t num_meshes;
	size_t num_items;
	size_t num_nodes;
	size_t num_triangles;
	size_t num_primitives;
	size_t num_vertices;
	size_t num_vertex_masks;
} rtki_build_counts;

struct rtk_build {

	rtki_build_counts counts;

	// Should the allocation be freed by RTK
	bool allocated;

	// Scene description passed to `rtk_start_build()`
	rtk_scene_desc desc;

	// Items to-be-inserted to the BVH, contains both primitives and triangles
	rtki_build_item *items;

	// BVH internal nodes, contains `a_num_nodes` nodes
	rtki_build_node *nodes;

	// Bit masks of global closed vertex indices that have been written
	uint32_t *vertex_written_mask;

	// Atomic count of in-flight tasks
	rtki_atomic_size a_tasks_left; 

	// Atomic count of internal BVH nodes
	rtki_atomic_size a_num_nodes; 

	// Task to submit when `a_tasks_left` reaches zero
	rtk_task next_task;

	// Threshold of a cheap too cheap to push to the queue
	double cheap_task_threshold;

	// AABB of the root BVH node
	rtk_bounds root_bounds;

	// Forked task states: Only one wide task is running concurrently
	union {
		struct {
			size_t count;
			rtki_setup_triangle_state state[RTKI_MAX_CONCURRENT_TASKS];
		} setup_triangles;
		struct {
			size_t count;
			rtki_setup_primitive_state state[RTKI_MAX_CONCURRENT_TASKS];
		} setup_primitives;
	} task;
};

struct rtk_build_ctx {

	// Build context
	rtk_build *build;

	// User-provided follow-up task queue
	rtk_task *followup_queue;
	size_t followup_queue_capacity;
	size_t followup_queue_num;
};

// Multi-threaded tasks
static void rtki_task_start(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_setup_triangles(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_start_primitives(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_setup_primitives(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_start_build_nodes(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_build_node(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_start_finalize_nodes(const rtk_task *task, rtk_build_ctx *ctx);
static void rtki_task_finalize_node(const rtk_task *task, rtk_build_ctx *ctx);

// Task costs
// TODO: Profile these to get reasonable estimates
#define RTKI_MS_SETUP_TRIANGLE 0.01
#define RTKI_MS_SETUP_PRIMITIVE 0.005

static void rtki_push_task_imp(rtk_build_ctx *ctx, rtk_task_fn *fn, size_t index, double cost, const char *name)
{
	rtk_assert(ctx->followup_queue_num < ctx->followup_queue_capacity);
	rtk_build *build = ctx->build;
	rtk_task *task = &ctx->followup_queue[ctx->followup_queue_num++];
	task->build = build;
	task->cost = cost;
	task->name = name;
	task->internal_fn = fn;
	task->internal_index = index;

	rtki_atomic_size_add(&build->a_tasks_left, 1);
}

// Always inlined version of `rtki_push_task_imp()` to make the indirect
// call direct when there is no follow-up queue or the task is cheap.
rtki_inline void rtki_push_task(rtk_build_ctx *ctx, rtk_task_fn *fn, size_t index, double cost, const char *name)
{
	// Run tasks inline if there's no space
	rtk_build *b = ctx->build;
	if (ctx->followup_queue_num == ctx->followup_queue_capacity || cost < b->cheap_task_threshold) {
		rtk_task task;
		task.build = ctx->build;
		task.cost = cost;
		task.name = name;
		task.internal_fn = fn;
		task.internal_index = index;
		fn(&task, ctx);
	} else {
		rtki_push_task_imp(ctx, fn, index, cost, name);
	}
}

static void rtki_task_start(const rtk_task *task, rtk_build_ctx *ctx)
{
	rtk_build *b = task->build;

	// Split the triangles into tasks, don't make tasks with too few triangles
	// and don't make too many tasks so we don't overflow the task queue and
	// execute them inline..
	size_t total_tris = b->counts.num_triangles;
	size_t task_tris = total_tris;
	if (total_tris > 0) {
		const size_t min_task_tris = 1024; 
		const size_t max_tasks = rtki_max_sz(ctx->followup_queue_capacity, RTKI_MAX_CONCURRENT_TASKS);
		if (max_tasks > 0 && total_tris > min_task_tris) {
			task_tris = rtki_max_sz(min_task_tris, (task_tris + max_tasks - 1) / max_tasks);
		}
		b->task.setup_triangles.count = (total_tris + task_tris - 1) / task_tris;
	}

	// Dispatch the tasks
	size_t task_ix = 0;
	for (size_t base = 0; base < total_tris; base += task_tris) {
		size_t num = rtki_min_sz(task_tris, total_tris - base);
		double cost = (double)num * RTKI_MS_SETUP_TRIANGLE;
		b->task.setup_triangles.state[task_ix].in_base = base;
		b->task.setup_triangles.state[task_ix].in_num = num;
		rtki_push_task(ctx, &rtki_task_setup_triangles, task_ix, cost, "setup triangles");
		task_ix++;
	}

	b->next_task.internal_fn = &rtki_task_start_primitives;
	b->next_task.name = "start primitives";
}

static void rtki_decode_indices(const rtk_index_desc *indices, uint32_t *dst, size_t offset, size_t count)
{
	// Prefer the user-provided callback function if there is one
	if (indices->func) {
		indices->func(indices->func_user, dst, offset, count);
		return;
	}

	// Fetch indices from the buffer if it exists
	if (indices->buffer.data) {

		// Expand type aliases
		rtk_datatype type = indices->buffer.type;
		if (type == RTK_DATATYPE_DEFAULT) {
			type = RTK_DATATYPE_U32;
		} else if (type == RTK_DATATYPE_SIZE_T) {
			switch (sizeof(size_t)) {
			case 2: type = RTK_DATATYPE_U16; break;
			case 4: type = RTK_DATATYPE_U32; break;
			case 8: type = RTK_DATATYPE_U64; break;
			}
		}

		if (type == RTK_DATATYPE_U16) {
			size_t stride = indices->buffer.stride ? indices->buffer.stride : sizeof(uint16_t) * 3;
			const void *data = (const char*)indices->buffer.data + offset * stride;
			const uint16_t *ix = (const uint16_t*)data;
			for (size_t i = 0; i < count; i++) {
				dst[0] = ix[0];
				dst[1] = ix[1];
				dst[2] = ix[2];
				ix += 3;
				dst += 3;
			}
		} else if (type == RTK_DATATYPE_U32) {
			size_t stride = indices->buffer.stride ? indices->buffer.stride : sizeof(uint32_t) * 3;
			const void *data = (const char*)indices->buffer.data + offset * stride;
			const uint32_t *ix = (const uint32_t*)data;
			for (size_t i = 0; i < count; i++) {
				dst[0] = ix[0];
				dst[1] = ix[1];
				dst[2] = ix[2];
				ix += 3;
				dst += 3;
			}
		} else if (type == RTK_DATATYPE_U64) {
			size_t stride = indices->buffer.stride ? indices->buffer.stride : sizeof(uint64_t) * 3;
			const void *data = (const char*)indices->buffer.data + offset * stride;
			const uint64_t *ix = (const uint64_t*)data;
			for (size_t i = 0; i < count; i++) {
				dst[0] = (uint32_t)ix[0];
				dst[1] = (uint32_t)ix[1];
				dst[2] = (uint32_t)ix[2];
				ix += 3;
				dst += 3;
			}
		} else {
			rtk_assert(0 && "Bad index type");
		}
	} else {
		// No index buffer: Just emit sequential indices
		for (size_t i = 0; i < count; i++) {
			uint32_t base = (uint32_t)(offset + i) * 3;
			dst[0] = base + 0;
			dst[1] = base + 1;
			dst[2] = base + 2;
			dst += 3;
		}
	}
}

static void rtki_decode_vertices(const rtk_vertex_desc *vertices, rtk_vec3 *dst, const uint32_t *indices, size_t count)
{
	// Prefer the user-provided callback function if there is one
	if (vertices->func) {
		vertices->func(vertices->func_user, dst, indices, count);
		return;
	}

	// If there's no callback a vertex buffer _must_ be provided
	rtk_assert(vertices->buffer.data);

	// Expand type aliases
	rtk_datatype type = vertices->buffer.type;
	if (type ==	RTK_DATATYPE_DEFAULT) {
		type = RTK_DATATYPE_REAL;
	}
	if (type == RTK_DATATYPE_REAL) {
		type = sizeof(rtk_real) == sizeof(double) ? RTK_DATATYPE_F64 : RTK_DATATYPE_F32;
	}

	if (type == RTK_DATATYPE_F32) {
		size_t stride = vertices->buffer.stride ? vertices->buffer.stride : sizeof(float) * 3;
		const char *data = (const char*)vertices->buffer.data;
		for (size_t i = 0; i < count; i++) {
			size_t base = i * 3;
			const float *a = (const float*)(data + indices[base + 0] * stride);
			const float *b = (const float*)(data + indices[base + 1] * stride);
			const float *c = (const float*)(data + indices[base + 2] * stride);
			dst[0].x = (rtk_real)a[0]; dst[0].y = (rtk_real)a[1]; dst[0].z = (rtk_real)a[2];
			dst[1].x = (rtk_real)b[0]; dst[1].y = (rtk_real)b[1]; dst[1].z = (rtk_real)b[2];
			dst[2].x = (rtk_real)c[0]; dst[2].y = (rtk_real)c[1]; dst[2].z = (rtk_real)c[2];
			dst += 3;
		}
	} else if (type == RTK_DATATYPE_F64) {
		size_t stride = vertices->buffer.stride ? vertices->buffer.stride : sizeof(double) * 3;
		const char *data = (const char*)vertices->buffer.data;
		for (size_t i = 0; i < count; i++) {
			size_t base = i * 3;
			const float *a = (const float*)(data + indices[base + 0] * stride);
			const float *b = (const float*)(data + indices[base + 1] * stride);
			const float *c = (const float*)(data + indices[base + 2] * stride);
			dst[0].x = (rtk_real)a[0]; dst[0].y = (rtk_real)a[1]; dst[0].z = (rtk_real)a[2];
			dst[1].x = (rtk_real)b[0]; dst[1].y = (rtk_real)b[1]; dst[1].z = (rtk_real)b[2];
			dst[2].x = (rtk_real)c[0]; dst[2].y = (rtk_real)c[1]; dst[2].z = (rtk_real)c[2];
			dst += 3;
		}
	} else {
		rtk_assert(0 && "Bad vertex type");
	}
}

static void rtki_task_setup_triangles(const rtk_task *task, rtk_build_ctx *ctx)
{
	rtk_build *b = task->build;

	rtki_setup_triangle_state *state = &b->task.setup_triangles.state[task->internal_index];

	size_t offset = state->in_base;
	size_t left = state->in_num;

	rtki_regbounds bounds;
	rtki_regbounds_reset(&bounds);

	rtki_build_item *item = b->items + offset;

	for (size_t mesh_ix = 0; mesh_ix < b->desc.num_meshes && left > 0; mesh_ix++) {

		// Scan meshes until we find one that our triangle range contains
		const rtk_mesh_desc *mesh = &b->desc.meshes[mesh_ix];
		if (offset >= mesh->num_triangles) {
			offset -= mesh->num_triangles;
			continue;
		}

		// Iterate vertices in fixed size chunks, first fetching a block of
		// indices, mapping those indices to vertices, and finally exporting
		// the resulting triangles as `rtk_build_item` items.
		size_t tris_left = mesh->num_triangles - offset;
		if (tris_left > left) tris_left = left;
		while (tris_left > 0) {
			size_t chunk_size = tris_left;
			if (chunk_size > 128) chunk_size = 128;
			uint32_t indices[128*3];
			rtk_vec3 vertices[128*3 + 1];

			rtki_decode_indices(&mesh->index, indices, offset, chunk_size);
			rtki_decode_vertices(&mesh->vertex, vertices, indices, chunk_size);

			for (size_t i = 0; i < chunk_size; i++) {
				size_t base = i * 3;
				rtki_reg3 v0 = rtki_load3x(&vertices[base + 0]);
				rtki_reg3 v1 = rtki_load3x(&vertices[base + 1]);
				rtki_reg3 v2 = rtki_load3x(&vertices[base + 2]);
				rtki_regbounds bb = { v0, v0 };
				rtki_bounds_add_point(&bb, v1);
				rtki_bounds_add_point(&bb, v2);
				rtki_bounds_add(&bounds, &bb);

				item->bounds.min = rtki_to_vec3(bb.min);
				item->bounds.max = rtki_to_vec3(bb.max);
				item->vertex[0].position = vertices[base + 0];
				item->vertex[0].index = indices[base + 0];
				item->vertex[1].position = vertices[base + 1];
				item->vertex[1].index = indices[base + 1];
				item->vertex[2].position = vertices[base + 2];
				item->vertex[2].index = indices[base + 2];
				item->mesh_prim_index = (uint32_t)mesh_ix;
				item->triangle_index = (uint32_t)(offset + i);
				item++;
			}

			offset += chunk_size;
			tris_left -= chunk_size;
		}

		offset = 0;
	}

	state->out_bounds.min = rtki_to_vec3(bounds.min);
	state->out_bounds.max = rtki_to_vec3(bounds.max);
}

static void rtki_task_start_primitives(const rtk_task *task, rtk_build_ctx *ctx)
{
	rtk_build *b = task->build;

	// Collect triangle bounding boxes
	rtki_regbounds bounds;
	rtki_regbounds_reset(&bounds);
	for (size_t i = 0; i < b->task.setup_triangles.count; i++) {
		rtki_setup_triangle_state *state = &b->task.setup_triangles.state[i];
		bounds.min = rtki_min3(bounds.min, rtki_load3x(&state->out_bounds.min));
		bounds.max = rtki_max3(bounds.max, rtki_loadx3(&state->out_bounds.max));
	}
	b->root_bounds.min = rtki_to_vec3(bounds.min);
	b->root_bounds.max = rtki_to_vec3(bounds.max);

	// Split the primitives into tasks, see how triangles are split into tasks
	// in `rtki_task_start()` for more explanation.
	size_t total_prims = b->counts.num_primitives;
	size_t task_prims = total_prims;
	if (total_prims > 0) {
		const size_t min_task_prims = 1024; 
		const size_t max_tasks = rtki_max_sz(ctx->followup_queue_capacity, RTKI_MAX_CONCURRENT_TASKS);
		if (max_tasks > 0 && total_prims > min_task_prims) {
			task_prims = rtki_max_sz(min_task_prims, (task_prims + max_tasks - 1) / max_tasks);
		}
		b->task.setup_primitives.count = (total_prims + task_prims - 1) / task_prims;
	}

	// Dispatch the tasks
	size_t task_ix = 0;
	for (size_t base = 0; base < total_prims; base += task_prims) {
		size_t num = rtki_min_sz(task_prims, total_prims - base);
		double cost = (double)num * RTKI_MS_SETUP_PRIMITIVE;
		b->task.setup_primitives.state[task_ix].in_base = base;
		b->task.setup_primitives.state[task_ix].in_num = num;
		rtki_push_task(ctx, &rtki_task_setup_primitives, task_ix, cost, "setup primitives");
		task_ix++;
	}

	// b->next_task.internal_fn = &rtki_task_start_primitives;
	b->next_task.internal_fn = NULL;
}

static void rtki_task_setup_primitives(const rtk_task *task, rtk_build_ctx *ctx)
{
	rtk_build *b = task->build;

	rtki_setup_primitive_state *state = &b->task.setup_primitives.state[task->internal_index];

	size_t offset = state->in_base;
	size_t num = state->in_num;

	rtki_regbounds bounds;
	rtki_regbounds_reset(&bounds);

	rtki_build_item *item = b->items + b->counts.num_triangles + offset;
	for (size_t i = 0; i < num; i++) {
		size_t prim_ix = offset + i;
		const rtk_primitive_desc *prim = b->desc.primitives + prim_ix;

		bounds.min = rtki_min3(bounds.min, rtki_load3x(&prim->bounds.min));
		bounds.max = rtki_max3(bounds.max, rtki_loadx3(&prim->bounds.max));

		item->bounds = prim->bounds;
		item->mesh_prim_index = (uint32_t)prim_ix;
		item->triangle_index = ~0u;
		item++;
	}
}

rtki_build_counts rtki_get_build_counts(const rtk_scene_desc *desc)
{
	rtki_build_counts counts = { 0 };

	counts.num_meshes = desc->num_meshes;
	for (size_t i = 0; i < desc->num_meshes; i++) {
		const rtk_mesh_desc *mesh = &desc->meshes[i];
		counts.num_triangles += mesh->num_triangles;
	}
	counts.num_primitives = desc->num_primitives;

	counts.num_items = counts.num_triangles + counts.num_primitives;
	counts.num_vertices = counts.num_triangles * 3;
	counts.num_vertex_masks = (counts.num_vertices + 31u) / 32u;

	// TODO: Better estimate
	counts.num_nodes = counts.num_items;

	size_t size = sizeof(rtk_build);
	size += sizeof(rtki_build_item) * counts.num_items;
	size += sizeof(rtki_build_node) * counts.num_nodes;
	size += sizeof(uint32_t) * counts.num_vertex_masks;
	counts.alloc_size = size;

	return counts;
}

// -- API

rtk_scene *rtk_build_scene(const rtk_scene_desc *desc)
{
	rtk_assert(desc);
	rtk_build *build = rtk_start_build(desc, NULL, NULL, 0);
	if (!build) return NULL;
	return rtk_finish_build(build, NULL, 0);
}

void rtk_free_scene(rtk_scene *scene)
{
	if (!scene) return;
	rtki_mem_free(scene, scene->header.size_in_bytes);
}

size_t rtk_get_build_size(const rtk_scene_desc *desc)
{
	rtk_assert(desc);
	rtki_build_counts counts = rtki_get_build_counts(desc);
	return counts.alloc_size;
}

rtk_build *rtk_start_build(const rtk_scene_desc *desc, rtk_task *opt_first_task, const void *opt_buffer, size_t opt_buffer_size)
{
	rtk_assert(desc);
	rtki_build_counts counts = rtki_get_build_counts(desc);

	// Either check that the user buffer is large enough to contain the build
	// operation or try to allocate a temporary buffer internally.
	rtk_build *build = NULL;
	if (opt_buffer) {
		if (opt_buffer_size < counts.alloc_size) return NULL;
		build = (rtk_build*)opt_buffer;
		build->allocated = false;
	} else {
		build = rtki_mem_alloc(counts.alloc_size);
		if (!build) return NULL;
		build->allocated = true;
	}

	// Resolve combined allocation pointers
	void *data_end = (char*)build + counts.alloc_size;
	build->items = (rtki_build_item*)(build + 1);
	build->nodes = (rtki_build_node*)(build->items + counts.num_items);
	build->vertex_written_mask = (uint32_t*)(build->nodes + counts.num_nodes);
	rtk_assert(build->vertex_written_mask + counts.num_vertex_masks == data_end);

	build->counts = counts;
	build->desc = *desc;

	// Consider the start task running
	build->a_tasks_left = 1;

	build->next_task.build = build;
	build->next_task.internal_index = 0;
	build->next_task.cost = 1000.0;

	rtk_task task_buf;
	rtk_task *first_task = opt_first_task ? opt_first_task : &task_buf;
	first_task->build = build;
	first_task->internal_fn = &rtki_task_start;
	first_task->internal_index = 0;
	first_task->cost = 1000.0;
	first_task->name = "start";
	if (!opt_first_task) {
		rtk_run_task(first_task, NULL, 0);
	}

	return build;
}

size_t rtk_run_task(const rtk_task *task, rtk_task *followup_queue, size_t followup_queue_size)
{
	rtk_build *b = task->build;

	// Setup a context with a follow-up queue to run tasks in
	rtk_build_ctx ctx;
	ctx.build = task->build;
	ctx.followup_queue = followup_queue;
	ctx.followup_queue_capacity = followup_queue_size;
	ctx.followup_queue_num = 0;

	// Run the task
	task->internal_fn(task, &ctx);

	// If we run out of currently running tasks execute the pending `next_task`
	// inline and see if it adds more tasks to the queue.
	size_t num_left = rtki_atomic_size_add(&b->a_tasks_left, -1) - 1;
	while (num_left == 0) {
		rtk_assert(ctx.followup_queue_num == 0);
		if (b->next_task.internal_fn) {
			rtki_atomic_size_add(&b->a_tasks_left, 1);
			if (ctx.followup_queue_num < ctx.followup_queue_capacity) {
				ctx.followup_queue[ctx.followup_queue_num++] = b->next_task;
				break;
			} else {
				rtk_task next_task = b->next_task;
				next_task.internal_fn(&next_task, &ctx);
				num_left = rtki_atomic_size_add(&b->a_tasks_left, -1) - 1;
			}
		} else {
			return RTK_BUILD_DONE;
		}
	}

	return ctx.followup_queue_num;
}

size_t rtk_get_scene_size(rtk_build *build)
{
	// TODO
	return 0;
}

rtk_scene *rtk_finish_build(rtk_build *build, const void *opt_buffer, size_t opt_buffer_size)
{
	// If we have allocated the build internally free it after we're done
	if (build->allocated) {
		rtki_mem_free(build, build->counts.alloc_size);
	}

	return NULL;
}

bool rtk_trace_ray(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit)
{
	// TODO
	return false;
}

#if 0
#include "rtk.h"

// Temp config
#define RTK_SSE 1
#define RTK_BVH_MAX_DEPTH 64
#define RTK_BVH_LEAF_MIN_ITEMS 4
#define RTK_BVH_LEAF_MAX_ITEMS 64


// -- Implementation

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#ifndef rtk_assert
#include <assert.h>
#define rtk_assert(x) assert(x)
#endif

#ifndef rtk_inline
#if defined(_MSC_VER)
	#define rtk_inline static __forceinline
#elif defined(__GNUC__)
	#define rtk_inline static inline __attribute__((always_inline))
#else
	#define rtk_inline static
#endif
#endif

#ifndef rtk_alloc
#if defined(_MSC_VER)
	#include <malloc.h>
	#define rtk_mem_alloc(size) _aligned_malloc(size, 64)
	#define rtk_mem_free(ptr, size) _aligned_free(ptr)
#elif defined(__GNUC__)
	#define rtk_mem_alloc(size) aligned_alloc(size, 64)
	#define rtk_mem_free(ptr, size) free(ptr)
#else
	// TODO: Manual alignment
	#define rtk_mem_alloc(size) malloc(size)
	#define rtk_mem_free(ptr, size) free(ptr)
#endif
#endif

#if defined(_MSC_VER)
	#include <intrin.h>
	#if defined(_M_X64)
		typedef uint64_t rtki_atomic_size;
		#define rtki_atomic_size_add(a, v) (size_t)_InterlockedExchangeAdd64((volatile long long*)(a), (long long)(v))
	#elif defined(_M_IX86)
		typedef uint32_t rtki_atomic_size;
		#define rtki_atomic_size_add(a, v) (size_t)_InterlockedExchangeAdd((volatile long*)(a), (long)(v))
	#else
		// TODO
	#endif
#endif

#ifndef RTK_SSE
#define RTK_SSE 0
#endif

// Tagged pointer
// [0:1] bool is_leaf;
// [0:64] uint64_t offset; // Offset from the beginning
typedef uint64_t rtk_ptr;

typedef struct {
	rtk_real bounds_x[2][4];
	rtk_real bounds_y[2][4];
	rtk_real bounds_z[2][4];
	rtk_ptr ptr[4];
} rtki_bvh_node; 

typedef struct rtki_bvh_leaf {
	// [0:6] num_triangles
	// [0:64] vertex_offset (aligned to 64 bytes)
	uint64_t triangle_info;
} rtki_bvh_leaf;

typedef struct {
	uint8_t v[3];
	uint8_t local_mesh_index;
	uint32_t triangle_index;
} rtki_leaf_triangle; 

typedef struct rtki_trace {
	const char *data;

	rtk_ray ray;
	rtk_hit hit;

	rtk_vec3 shear_origin;
	rtk_vec3 shear;

	uint32_t sign_mask;
	uint32_t shear_x;
	uint32_t shear_y;
	uint32_t shear_z;

} rtki_trace;

rtk_inline size_t rtki_align_up_sz(size_t v, size_t align) {
	return v + ((size_t)-(intptr_t)v & (align - 1));
}

#if RTK_SSE

// -- SSE backend

#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>

#if 1
	#define RTK_MM_LOAD_F32(ptr) _mm_loadu_ps((const float*)(ptr))
	#define RTK_MM_LOAD_SI128(ptr) _mm_loadu_si128((const __m128i*)(ptr))
#else
	#define RTK_MM_LOAD_F32(ptr) _mm_load_ps((const float*)(ptr))
	#define RTK_MM_LOAD_SI128(ptr) _mm_load_si128((const __m128i*)(ptr))
#endif

typedef __m128 rtki_reg3;
rtk_inline rtki_reg3 rtki_load3(const rtk_vec3 *v) { return _mm_setr_ps(v->x, v->y, v->z, v->z); }
rtk_inline rtki_reg3 rtki_load3x(const rtk_vec3 *v) {
	__m128 t = _mm_loadu_ps(&v->x);
	return _mm_shuffle_ps(t, t, _MM_SHUFFLE(2,2,1,0));
}
rtk_inline rtki_reg3 rtki_loadx3(const rtk_vec3 *v) {
	__m128 t = _mm_loadu_ps(&v->x - 1);
	return _mm_shuffle_ps(t, t, _MM_SHUFFLE(3,3,2,1));
}
#define rtki_broadcast3(x) _mm_set1_ps((x))
#define rtki_add3(a, b) _mm_add_ps((a), (b))
#define rtki_sub3(a, b) _mm_sub_ps((a), (b))
#define rtki_mul3(a, b) _mm_mul_ps((a), _mm_set_ss(b))
#define rtki_min3(a, b) _mm_min_ps((a), (b))
#define rtki_max3(a, b) _mm_max_ps((a), (b))
#define rtki_abs3(a) _mm_andnot_ps(_mm_set1_ps(-0.0f), (a))
#define rtki_x3(a) _mm_cvtss_f32((a))
rtk_inline float rtki_y3(rtki_reg3 r) { return _mm_cvtss_f32(_mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,1))); }
rtk_inline float rtki_z3(rtki_reg3 r) { return _mm_cvtss_f32(_mm_movehl_ps(r, r)); }
rtk_inline rtk_vec3 rtki_to_vec3(rtki_reg3 r) {
	rtk_vec3 v = { rtki_x3(r), rtki_y3(r), rtki_z3(r) };
	return v;
}
rtk_inline rtk_real rtki_maxcomp3(rtki_reg3 r) {
	__m128 tmp = _mm_max_ps(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,1)));
	return _mm_cvtss_f32(_mm_max_ps(tmp, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3,2,1,2))));
}
rtk_inline uint32_t rtki_signmask3(rtki_reg3 r) {
	return (uint32_t)_mm_movemask_ps(r) & 0x7u;
}

#define RTK_MM_BROADCAST(reg, lane) _mm_shuffle_ps((reg), (reg), _MM_SHUFFLE(lane, lane, lane, lane))
#define RTK_MM_MIN4(a, b, c, d) _mm_min_ps(_mm_min_ps((a), (b)), _mm_min_ps((c), (d)))
#define RTK_MM_MAX4(a, b, c, d) _mm_max_ps(_mm_max_ps((a), (b)), _mm_max_ps((c), (d)))
#define RTK_MM_SUBMUL(a, b, c) _mm_mul_ps(_mm_sub_ps(RTK_MM_LOAD_F32((a)), (b)), (c))
#define RTK_MM_GET_F32(a, ix) _mm_cvtss_f32(_mm_shuffle_ps((a), (a), _MM_SHUFFLE(3,2,1,ix)))
#define RTK_MM_GET_U32(a, ix) ((uint32_t)_mm_cvtsi128_si32(_mm_shuffle_epi32((a), _MM_SHUFFLE(3,2,1,ix))))
#define RTK_MM_RCP(a) _mm_div_ps(_mm_set1_ps(1.0f), (a))

static const uint32_t shuf_u32_tab[4] = {
	0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
};

static void rtki_leaf_traverse_sse(rtki_trace *rt, rtk_ptr leaf_ptr)
{
	const char *const data = rt->data;
	rtk_assert((leaf_ptr & 1) == 1);

	const rtki_bvh_leaf *const leaf = (const rtki_bvh_leaf*)(data + (leaf_ptr^1));
	const uint64_t triangle_info = leaf->triangle_info;
	const size_t num_tris = triangle_info & 0x3f;
	const size_t num_tris_aligned = (num_tris + 3) & ~3u;

	const rtki_leaf_triangle *tris = (const rtki_leaf_triangle*)(leaf + 1);
	const uint32_t *mesh_indices = (const uint32_t*)(tris + num_tris_aligned);
	const rtk_vertex *verts = (const rtk_vertex*)(data + (triangle_info & ~0x3f));

	const __m128 shear_origin_x = _mm_set1_ps(rt->shear_origin.x);
	const __m128 shear_origin_y = _mm_set1_ps(rt->shear_origin.y);
	const __m128 shear_origin_z = _mm_set1_ps(rt->shear_origin.z);

	const __m128 shear_x = _mm_set1_ps(rt->shear.x);
	const __m128 shear_y = _mm_set1_ps(rt->shear.y);
	const __m128 shear_z = _mm_set1_ps(rt->shear.z);

	// Setup the _mm_shuffle_epi8() tables
	const __m128i shear_shuf = _mm_setr_epi32((int)shuf_u32_tab[rt->shear_x],
		(int)shuf_u32_tab[rt->shear_y], (int)shuf_u32_tab[rt->shear_z], -1);

	// Ray distance limits
	const __m128 min_t = _mm_set1_ps(rt->ray.min_t);
	__m128 max_t = _mm_set1_ps(rt->hit.t);

	const rtki_leaf_triangle *tris_end = tris + num_tris_aligned;
	for (; tris != tris_end; tris += 4) {

		// Load AOS triangle data (XYZ + vertex index)
		const __m128i ai0 = RTK_MM_LOAD_SI128(verts + tris[0].v[0]);
		const __m128i bi0 = RTK_MM_LOAD_SI128(verts + tris[1].v[0]);
		const __m128i ci0 = RTK_MM_LOAD_SI128(verts + tris[2].v[0]);
		const __m128i di0 = RTK_MM_LOAD_SI128(verts + tris[3].v[0]);

		const __m128i ai1 = RTK_MM_LOAD_SI128(verts + tris[0].v[1]);
		const __m128i bi1 = RTK_MM_LOAD_SI128(verts + tris[1].v[1]);
		const __m128i ci1 = RTK_MM_LOAD_SI128(verts + tris[2].v[1]);
		const __m128i di1 = RTK_MM_LOAD_SI128(verts + tris[3].v[1]);

		const __m128i ai2 = RTK_MM_LOAD_SI128(verts + tris[0].v[2]);
		const __m128i bi2 = RTK_MM_LOAD_SI128(verts + tris[1].v[2]);
		const __m128i ci2 = RTK_MM_LOAD_SI128(verts + tris[2].v[2]);
		const __m128i di2 = RTK_MM_LOAD_SI128(verts + tris[3].v[2]);

		// Permute the coordinate axes to shear space

		const __m128 a0 = _mm_castsi128_ps(_mm_shuffle_epi8(ai0, shear_shuf));
		const __m128 b0 = _mm_castsi128_ps(_mm_shuffle_epi8(bi0, shear_shuf));
		const __m128 c0 = _mm_castsi128_ps(_mm_shuffle_epi8(ci0, shear_shuf));
		const __m128 d0 = _mm_castsi128_ps(_mm_shuffle_epi8(di0, shear_shuf));
		const __m128 a1 = _mm_castsi128_ps(_mm_shuffle_epi8(ai1, shear_shuf));
		const __m128 b1 = _mm_castsi128_ps(_mm_shuffle_epi8(bi1, shear_shuf));
		const __m128 c1 = _mm_castsi128_ps(_mm_shuffle_epi8(ci1, shear_shuf));
		const __m128 d1 = _mm_castsi128_ps(_mm_shuffle_epi8(di1, shear_shuf));
		const __m128 a2 = _mm_castsi128_ps(_mm_shuffle_epi8(ai2, shear_shuf));
		const __m128 b2 = _mm_castsi128_ps(_mm_shuffle_epi8(bi2, shear_shuf));
		const __m128 c2 = _mm_castsi128_ps(_mm_shuffle_epi8(ci2, shear_shuf));
		const __m128 d2 = _mm_castsi128_ps(_mm_shuffle_epi8(di2, shear_shuf));

		// Tranpose the triangle coordinates/indices to SOA

		__m128 t0, t1, vx, vy, vz;

		t0 = _mm_unpacklo_ps(a0, b0); // XX YY
		t1 = _mm_unpacklo_ps(c0, d0); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a0, b0); // ZZ II
		t1 = _mm_unpackhi_ps(c0, d0); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v0x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v0y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v0z = _mm_sub_ps(vz, shear_origin_z);

		t0 = _mm_unpacklo_ps(a1, b1); // XX YY
		t1 = _mm_unpacklo_ps(c1, d1); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a1, b1); // ZZ II
		t1 = _mm_unpackhi_ps(c1, d1); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v1x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v1y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v1z = _mm_sub_ps(vz, shear_origin_z);

		t0 = _mm_unpacklo_ps(a2, b2); // XX YY
		t1 = _mm_unpacklo_ps(c2, d2); // XX YY
		vx = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		vy = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		t0 = _mm_unpackhi_ps(a2, b2); // ZZ II
		t1 = _mm_unpackhi_ps(c2, d2); // ZZ II
		vz = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		const __m128 v2x = _mm_sub_ps(vx, shear_origin_x);
		const __m128 v2y = _mm_sub_ps(vy, shear_origin_y);
		const __m128 v2z = _mm_sub_ps(vz, shear_origin_z);

		// Transform the triangle coordinates to sheared ray space

		const __m128 x0 = _mm_add_ps(v0x, _mm_mul_ps(shear_x, v0z));
		const __m128 y0 = _mm_add_ps(v0y, _mm_mul_ps(shear_y, v0z));
		const __m128 z0 = _mm_mul_ps(shear_z, v0z);
		const __m128 x1 = _mm_add_ps(v1x, _mm_mul_ps(shear_x, v1z));
		const __m128 y1 = _mm_add_ps(v1y, _mm_mul_ps(shear_y, v1z));
		const __m128 z1 = _mm_mul_ps(shear_z, v1z);
		const __m128 x2 = _mm_add_ps(v2x, _mm_mul_ps(shear_x, v2z));
		const __m128 y2 = _mm_add_ps(v2y, _mm_mul_ps(shear_y, v2z));
		const __m128 z2 = _mm_mul_ps(shear_z, v2z);

		// Calculate the edge functions u,v,w. If any of these functions is
		// exactly zero recalculate them all using double precision to
		// guarantee watertight intersections.

		__m128 u = _mm_sub_ps(_mm_mul_ps(x1, y2), _mm_mul_ps(y1, x2));
		__m128 v = _mm_sub_ps(_mm_mul_ps(x2, y0), _mm_mul_ps(y2, x0));
		__m128 w = _mm_sub_ps(_mm_mul_ps(x0, y1), _mm_mul_ps(y0, x1));
		const __m128 zero = _mm_setzero_ps();
		const __m128 zero_u = _mm_cmpeq_ps(u, zero);
		const __m128 zero_v = _mm_cmpeq_ps(v, zero);
		const __m128 zero_w = _mm_cmpeq_ps(w, zero);
		const __m128 any_zero = _mm_or_ps(_mm_or_ps(zero_u, zero_v), zero_w);
		if (_mm_movemask_ps(any_zero) != 0) {
			__m128d xd0 = _mm_cvtps_pd(x0);
			__m128d yd0 = _mm_cvtps_pd(y0);
			__m128d xd1 = _mm_cvtps_pd(x1);
			__m128d yd1 = _mm_cvtps_pd(y1);
			__m128d xd2 = _mm_cvtps_pd(x2);
			__m128d yd2 = _mm_cvtps_pd(y2);

			__m128d ud = _mm_sub_pd(_mm_mul_pd(xd1, yd2), _mm_mul_pd(yd1, xd2));
			__m128d vd = _mm_sub_pd(_mm_mul_pd(xd2, yd0), _mm_mul_pd(yd2, xd0));
			__m128d wd = _mm_sub_pd(_mm_mul_pd(xd0, yd1), _mm_mul_pd(yd0, xd1));

			u = _mm_cvtpd_ps(ud);
			v = _mm_cvtpd_ps(vd);
			w = _mm_cvtpd_ps(wd);

			xd0 = _mm_cvtps_pd(_mm_movehl_ps(x0, x0));
			yd0 = _mm_cvtps_pd(_mm_movehl_ps(y0, y0));
			xd1 = _mm_cvtps_pd(_mm_movehl_ps(x1, x1));
			yd1 = _mm_cvtps_pd(_mm_movehl_ps(y1, y1));
			xd2 = _mm_cvtps_pd(_mm_movehl_ps(x2, x2));
			yd2 = _mm_cvtps_pd(_mm_movehl_ps(y2, y2));

			ud = _mm_sub_pd(_mm_mul_pd(xd1, yd2), _mm_mul_pd(yd1, xd2));
			vd = _mm_sub_pd(_mm_mul_pd(xd2, yd0), _mm_mul_pd(yd2, xd0));
			wd = _mm_sub_pd(_mm_mul_pd(xd0, yd1), _mm_mul_pd(yd0, xd1));

			u = _mm_movelh_ps(u, _mm_cvtpd_ps(ud));
			v = _mm_movelh_ps(v, _mm_cvtpd_ps(vd));
			w = _mm_movelh_ps(w, _mm_cvtpd_ps(wd));
		}

		// Only if all u,v,w are either negative or positive we hit the triangle,
		// so we can bail out if both signs are represented
		const __m128 neg = _mm_cmplt_ps(_mm_min_ps(_mm_min_ps(u, v), w), zero);
		const __m128 pos = _mm_cmpgt_ps(_mm_max_ps(_mm_max_ps(u, v), w), zero);
		const __m128 bad_sign = _mm_and_ps(neg, pos);
		int bad_sign_mask = _mm_movemask_ps(bad_sign);
		if (bad_sign_mask == 0xf) continue;

		__m128 det = _mm_add_ps(_mm_add_ps(u, v), w);
		__m128 rcp_det = RTK_MM_RCP(det);

		__m128 z = _mm_mul_ps(u, z0);
		z = _mm_add_ps(z, _mm_mul_ps(v, z1));
		z = _mm_add_ps(z, _mm_mul_ps(w, z2));

		__m128 t = _mm_mul_ps(z, rcp_det);
		__m128 good = _mm_and_ps(_mm_cmpgt_ps(t, min_t), _mm_cmplt_ps(t, max_t));

		// Scalar loop over hit triangles
		int good_mask = _mm_movemask_ps(good);
		good_mask &= ~bad_sign_mask;
		if (good_mask != 0) {
			RTK_ALIGN16 float ts[4], us[4], vs[4];

			_mm_store_ps(ts, t);
			_mm_store_ps(us, _mm_mul_ps(u, rcp_det));
			_mm_store_ps(vs, _mm_mul_ps(v, rcp_det));

			do {
				uint32_t lane_ix;
				RTK_FIRSTBIT4(lane_ix, good_mask);

				float lane_t = ts[lane_ix];
				if (lane_t < rt->hit.t) {
					const rtki_leaf_triangle *tri = &tris[lane_ix];
					rt->hit.t = lane_t;
					rt->hit.u = us[lane_ix];
					rt->hit.v = vs[lane_ix];
					rt->hit.vertex[0] = verts[tri->v[0]];
					rt->hit.vertex[1] = verts[tri->v[1]];
					rt->hit.vertex[2] = verts[tri->v[2]];
					rt->hit.mesh_index = mesh_indices[tri->local_mesh_index];
					rt->hit.triangle_index = tri->triangle_index;
					max_t = _mm_set1_ps(lane_t);
				}

				good_mask &= good_mask - 1;
			} while (good_mask != 0);
		}
	}
}

static void rtki_bvh_traverse_sse(rtki_trace *rt, rtk_ptr root_ptr)
{
	const char *const data = rt->data;
	rtk_assert((root_ptr & 1) == 0);

	// Traversal stack: The top node is kept separate in registers,
	// the stack only contains the pushed t/ptr values. Initialize
	// the stack so that it's safe to pop once without checking 
	// that the stack is exhausted.
	float top_t = -RTK_INF;
	rtk_ptr top_ptr = root_ptr;
	float stack_t[RTK_BVH_MAX_DEPTH + 1];
	uint64_t stack_ptr[RTK_BVH_MAX_DEPTH + 1];
	uint32_t stack_depth = 2;
	stack_t[1] = stack_t[0] = RTK_INF;
	stack_ptr[1] = stack_ptr[0] = 0;

	const __m128 pos_inf = _mm_set1_ps(3.402823e+38f);

	const __m128 ray_origin = _mm_loadu_ps(&rt->ray.origin.x);
	const __m128 ray_rcp_dir = RTK_MM_RCP(_mm_loadu_ps(&rt->ray.direction.x));
	const __m128 origin_x = RTK_MM_BROADCAST(ray_origin, 0);
	const __m128 origin_y = RTK_MM_BROADCAST(ray_origin, 1);
	const __m128 origin_z = RTK_MM_BROADCAST(ray_origin, 2);
	const __m128 rcp_dir_x = RTK_MM_BROADCAST(ray_rcp_dir, 0);
	const __m128 rcp_dir_y = RTK_MM_BROADCAST(ray_rcp_dir, 1);
	const __m128 rcp_dir_z = RTK_MM_BROADCAST(ray_rcp_dir, 2);
	const __m128 ray_min_t = _mm_load1_ps(&rt->ray.min_t);

	const uint32_t sign_mask = rt->sign_mask;
	const uint32_t sign_x = sign_mask & 1;
	const uint32_t sign_y = sign_mask >> 1 & 1;
	const uint32_t sign_z = sign_mask >> 2;

	const __m128 tag_mask = _mm_castsi128_ps(_mm_set1_epi32(0x3));
	const __m128 tag_indices = _mm_castsi128_ps(_mm_setr_epi32(0, 1, 2, 3));

	for (;;) {
		const float ray_hit_t = rt->hit.t;

		// Pop nodes while our current geometry hit distance is less
		// than the current top BVH hit distance.
		while (top_t >= ray_hit_t) {
			rtk_assert(stack_depth > 0);
			if (--stack_depth == 0) return;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
		}

		// Now we have a valid top BVH node! If it's a leaf traverse the
		// primitives and pop the stack and `continue` to return to it.
		if ((top_ptr & 1) != 0) {
			rtki_leaf_traverse_sse(rt, top_ptr);
			--stack_depth;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
			continue;
		}

		// The top BVH node is an internal node: Intersect the ray against the
		// axis+aligned planes and find the intersection distances `min_t` and `max_t`.
		// We want to traverse to the child BVH node if all:
		// (1) `min_t <= max_t`: The ray intersects the AABB
		// (2) `min_t <= ray_hit_t`: The AABB intersection distance is closer than ray hit/max
		// (3) `max_t >= ray_min_t`: The far AABB distance is further than the ray min
		// We can fold all these tests into a single expression:
		// `max(min_t, ray_min_t) <= min(max_t, ray_hit_t)`
		const rtki_bvh_node *node = (const rtki_bvh_node*)(data + (size_t)top_ptr);
		const __m128 min_x = RTK_MM_SUBMUL(node->bounds_x[sign_x    ], origin_x, rcp_dir_x);
		const __m128 max_x = RTK_MM_SUBMUL(node->bounds_x[sign_x ^ 1], origin_x, rcp_dir_x);
		const __m128 min_y = RTK_MM_SUBMUL(node->bounds_y[sign_y    ], origin_y, rcp_dir_y);
		const __m128 max_y = RTK_MM_SUBMUL(node->bounds_y[sign_y ^ 1], origin_y, rcp_dir_y);
		const __m128 min_z = RTK_MM_SUBMUL(node->bounds_z[sign_z    ], origin_z, rcp_dir_z);
		const __m128 max_z = RTK_MM_SUBMUL(node->bounds_z[sign_z ^ 1], origin_z, rcp_dir_z);
		const __m128 min_t = RTK_MM_MAX4(min_x, min_y, min_z, ray_min_t);
		const __m128 max_t = RTK_MM_MIN4(max_x, max_y, max_z, _mm_set1_ps(ray_hit_t));


		// Make a mask of nodes to traverse, replace t values for inactive lanes with
		// infinity and count the number of hit node bits.
		const __m128 mask = _mm_cmple_ps(min_t, max_t);
		const __m128 ts = RTK_MM_BLEND(pos_inf, min_t, mask);
		uint32_t mask_bits = (uint32_t)_mm_movemask_ps(mask);
		uint32_t num_bits = RTK_POPCOUNT4(mask_bits);

		if (mask_bits == 0) {
			// Hit nothing: Pop the BVH node stack
			rtk_assert(stack_depth > 0);
			--stack_depth;
			top_t = stack_t[stack_depth];
			top_ptr = stack_ptr[stack_depth];
		} else if (num_bits == 1) {
			// Hit a single node: Replace the top node with the minimum distance
			// and first (only) found bit index.
			uint32_t lowest_bit;
			RTK_FIRSTBIT4(lowest_bit, mask_bits);
			const __m128 tmp = _mm_min_ps(ts, _mm_shuffle_ps(ts, ts, _MM_SHUFFLE(2,3,0,1))); 
			top_t = _mm_cvtss_f32(_mm_min_ps(tmp, _mm_movehl_ps(tmp, tmp)));
			top_ptr = node->ptr[lowest_bit];
		} else {
			// Hit 2-4 nodes: Tag the hit distances with indices (two lowest bits)
			// and sort them inside the XMM registers using a sorting network:
			// t_tag[0] --v--- t_lo[01] - t_1[0] --v--- t_lo[01] - t_2[0] ----- t_lo/hi[0] - t_tag_sorted[0]
			// t_tag[1] --^--- t_hi[01] - t_1[2] --|v-- t_lo[23] - t_2[1] --v--  t_lo[12]  - t_tag_sorted[1]
			// t_tag[2] ---v-- t_lo[23] - t_1[1] --^|-- t_hi[01] - t_2[2] --^--  t_hi[12]  - t_tag_sorted[2]
			// t_tag[3] ---^-- t_hi[23] - t_1[3] ---^-- t_hi[23] - t_2[3] ----- t_lo/hi[3] - t_tag_sorted[3]
			const __m128 t_tag = _mm_or_ps(_mm_andnot_ps(tag_mask, ts), tag_indices);

			__m128 t_swap = _mm_shuffle_ps(t_tag, t_tag, _MM_SHUFFLE(2,3,0,1));
			__m128 t_lo = _mm_min_ps(t_tag, t_swap);
			__m128 t_hi = _mm_max_ps(t_tag, t_swap);
			const __m128 t_1 = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(2,0,2,0));

			t_swap = _mm_shuffle_ps(t_1, t_1, _MM_SHUFFLE(2,3,0,1));
			t_lo = _mm_min_ps(t_1, t_swap);
			t_hi = _mm_max_ps(t_1, t_swap);
			const __m128 t_2 = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(2,0,2,0));

			t_swap = _mm_shuffle_ps(t_2, t_2, _MM_SHUFFLE(3,1,2,0));
			t_lo = _mm_min_ps(t_2, t_swap);
			t_hi = _mm_max_ps(t_2, t_swap);

			const __m128 t_tag_sorted = _mm_shuffle_ps(t_lo, t_hi, _MM_SHUFFLE(3,2,1,0));

			// Clear the tag bits from the sorted values, this means we might
			// unnecessarily enter nodes that are 4 bits away from the max distance.
			const __m128 t_sorted = _mm_andnot_ps(tag_mask, t_tag_sorted);
			const __m128i tag_sorted = _mm_castps_si128(_mm_and_ps(tag_mask, t_tag_sorted));

			// Traverse the top hit directly and push the rest to the stack
			stack_depth += num_bits - 1;
			float *dst_t = stack_t + stack_depth;

			rtk_ptr *dst_ptr = stack_ptr + stack_depth;
			top_t = RTK_MM_GET_F32(t_sorted, 0);
			top_ptr = node->ptr[RTK_MM_GET_U32(tag_sorted, 0)];

			dst_t[0] = RTK_MM_GET_F32(t_sorted, 1);
			dst_ptr[0] = node->ptr[RTK_MM_GET_U32(tag_sorted, 1)];
			if (num_bits >= 3) {
				dst_t[1] = RTK_MM_GET_F32(t_sorted, 2);
				dst_ptr[1] = node->ptr[RTK_MM_GET_U32(tag_sorted, 2)];
				if (num_bits == 4) {
					dst_t[2] = RTK_MM_GET_F32(t_sorted, 3);
					dst_ptr[2] = node->ptr[RTK_MM_GET_U32(tag_sorted, 3)];
				}
			}
		}
	}
}

#endif

bool rtk_trace_ray(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit)
{
	rtki_trace rt;
	rt.data = (const char*)scene;
	rt.ray = *ray;
	rt.hit.t = ray->max_t;

	rtki_reg3 dir = rtki_loadx3(&ray->direction);
	rtki_reg3 abs = rtki_abs3(dir);
	rtk_real max_comp = rtki_maxcomp3(abs);
	uint32_t shear_z = rtki_x3(abs) == max_comp ? 0 : rtki_y3(abs) == max_comp ? 1 : 2;
	uint32_t shear_x = (shear_z + 1) % 3u;
	uint32_t shear_y = (shear_z + 2) % 3u;
	uint32_t sign_mask = rtki_signmask3(dir);

	rt.shear_x = shear_x;
	rt.shear_y = shear_y;
	rt.shear_z = shear_z;
	rt.shear.x = -ray->direction.v[shear_x] / ray->direction.v[shear_z];
	rt.shear.y = -ray->direction.v[shear_y] / ray->direction.v[shear_z];
	rt.shear.z = 1.0f / ray->direction.v[shear_z];
	rt.shear_origin.x = ray->origin.v[shear_x];
	rt.shear_origin.y = ray->origin.v[shear_y];
	rt.shear_origin.z = ray->origin.v[shear_z];
	rt.sign_mask = sign_mask;

	rtki_bvh_traverse_sse(&rt, 128);

	if (rt.hit.t < ray->max_t) {
		*hit = rt.hit;
		return true;
	} else {
		return false;
	}
}

bool rtk_trace_ray_filter(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit, rtk_filter_fn *filter, void *filter_user)
{
	return true;
}

// -- Scene building

#ifndef RTK_BUILD_SPLITS
#define RTK_BUILD_SPLITS 32
#endif

#ifndef RTK_MAX_CONCURRENT_TASKS
#define RTK_MAX_CONCURRENT_TASKS 128
#endif

typedef struct {
	rtki_reg3 min, max;
} rtki_bounds3;

typedef struct {
	rtk_vec3 min, max;
	rtk_vertex v[3];
	uint32_t mesh_index;
	uint32_t triangle_index;
	uint8_t v_ix[3];
} rtki_build_item;

typedef struct {
	rtk_vec3 min, max;
	size_t item_begin;
	size_t item_count;
	size_t child_index;
	size_t vertex_offset;
	uint16_t depth;
} rtki_build_node;

typedef struct {
	rtki_bounds3 bounds;       // < Bounds of all the items in this bucket
	rtki_bounds3 bounds_right; // < Cumulative bounds for all the buckets from the right
	uint32_t num;              // < Number of items in the bucket
} rtki_build_bucket;

typedef struct {
	size_t base;
	size_t num;
	rtk_vec3 min, max;
} rtki_triangle_task_state;

struct rtk_build {
	size_t alloc_size;

	rtk_scene_desc desc;

	size_t num_triangles;

	rtki_build_item *items;
	rtki_build_node *nodes;

	rtki_atomic_size a_depth_num_nodes[RTK_BVH_MAX_DEPTH];
	rtki_atomic_size a_nodes_allocated;
	rtki_atomic_size a_tasks_left;
	rtki_atomic_size a_vertex_count;
	rtki_atomic_size a_leaf_size;

	size_t depth_node_offset[RTK_BVH_MAX_DEPTH];

	double min_task_cost;

	rtk_real sah_item_cost;
	rtk_real sah_split_cost;
	size_t max_leaf_items;

	double bvh_task_item_cost;
	double finalize_task_item_cost;
	double triangle_task_item_cost;

	uint64_t node_data_offset;
	uint64_t leaf_data_offset;
	uint64_t vertex_data_offset;

	rtki_bvh_node *final_nodes;
	rtki_bvh_leaf *final_leaves;
	rtk_vertex *final_vertices;

	uint32_t *vertex_written_mask;
	size_t num_vertex_masks;

	size_t final_leaf_offset;
	size_t final_vertex_offset;

	rtk_task next_task;

	union {
		struct {
			size_t count;
			rtki_triangle_task_state state[RTK_MAX_CONCURRENT_TASKS];
		} triangle;
	} task;
};

struct rtk_task_ctx {
	rtk_build *build;
	rtk_task *queue;
	size_t queue_capacity;
	size_t queue_num;
};

static void rtki_build_log(rtk_build *b, const char *fmt, ...)
{
	if (b->desc.log_fn) {
		char buf[256];
		va_list args;
		va_start(args, fmt);
		vsnprintf(buf, sizeof(buf), fmt, args);
		b->desc.log_fn(b->desc.log_user, b, buf);
		va_end(args);
	}
}

// Tasks

static void rtki_push_task(rtk_task_ctx *ctx, rtk_task_fn *fn, size_t index, uintptr_t arg, double cost)
{
	rtk_build *build = ctx->build;
	rtk_task *task = &ctx->queue[ctx->queue_num++];
	task->build = build;
	task->fn = fn;
	task->index = index;
	task->arg = arg;
	task->cost = cost;
	rtki_atomic_size_add(&build->a_tasks_left, 1);
}

// Bounds

rtk_inline void rtki_bounds_reset(rtki_bounds3 *bounds) {
	bounds->min = rtki_broadcast3(+RTK_INF);
	bounds->max = rtki_broadcast3(-RTK_INF);
}

rtk_inline void rtki_bounds_add(rtki_bounds3 *dst, const rtki_bounds3 *a) {
	dst->min = rtki_min3(dst->min, a->min);
	dst->max = rtki_max3(dst->max, a->max);
}

rtk_inline void rtki_bounds_add_point(rtki_bounds3 *dst, rtki_reg3 a) {
	dst->min = rtki_min3(dst->min, a);
	dst->max = rtki_max3(dst->max, a);
}

rtk_inline rtk_real rtki_bounds_area(const rtki_bounds3 *bounds) {
	rtki_reg3 d = rtki_sub3(bounds->max, bounds->min);
	rtk_real x = rtki_x3(d), y = rtki_y3(d), z = rtki_z3(d);
	return 2.0f * (x*y + y*z + z*x);
}

// BVH

typedef int (*rtki_sort_fn)(const void *a, const void *b);

static int rtki_item_sort_axis_x(const void *a, const void *b) {
	rtk_real mid_a = ((const rtki_build_item*)a)->min.x + ((const rtki_build_item*)a)->max.x;
	rtk_real mid_b = ((const rtki_build_item*)b)->min.x + ((const rtki_build_item*)b)->max.x;
	return mid_a < mid_b ? -1 : +1;
}

static int rtki_item_sort_axis_y(const void *a, const void *b) {
	rtk_real mid_a = ((const rtki_build_item*)a)->min.y + ((const rtki_build_item*)a)->max.y;
	rtk_real mid_b = ((const rtki_build_item*)b)->min.y + ((const rtki_build_item*)b)->max.y;
	return mid_a < mid_b ? -1 : +1;
}

static int rtki_item_sort_axis_z(const void *a, const void *b) {
	rtk_real mid_a = ((const rtki_build_item*)a)->min.z + ((const rtki_build_item*)a)->max.z;
	rtk_real mid_b = ((const rtki_build_item*)b)->min.z + ((const rtki_build_item*)b)->max.z;
	return mid_a < mid_b ? -1 : +1;
}

static rtki_sort_fn rtki_item_sort_axis_fn[3] = {
	&rtki_item_sort_axis_x,
	&rtki_item_sort_axis_y,
	&rtki_item_sort_axis_z,
};

static void rtki_task_build_node(const rtk_task *task, rtk_task_ctx *ctx);

static void rtki_build_node_leaf(rtk_task_ctx *ctx, rtki_build_node *node)
{
	rtk_build *b = ctx->build;

	rtk_assert(node->item_count <= b->max_leaf_items);
	node->child_index = SIZE_MAX;
	node->vertex_offset = SIZE_MAX;

	rtki_reg3 size = rtki_sub3(rtki_load3x(&node->max), rtki_loadx3(&node->max));
	rtk_real max_comp = rtki_maxcomp3(size);
	int axis = rtki_x3(size) == max_comp ? 0 : rtki_y3(size) == max_comp ? 1 : 2;

	// Sort items by the largest axis
	rtki_build_item *items = b->items + node->item_begin;
	qsort(items, node->item_count, sizeof(rtki_build_item), rtki_item_sort_axis_fn[axis]);

	// Collect unique meshes
	uint32_t unique_meshes[RTK_BVH_LEAF_MAX_ITEMS];
	uint32_t num_unique_meshes = 0;

	uint32_t num_tris = 0;
	for (size_t i = 0; i < node->item_count; i++) {
		rtki_build_item *item = &items[i];
		uint32_t mesh_ix = item->mesh_index;
		if (mesh_ix == ~0u) continue;
		num_tris++;

		// O(n^2) but n is bounded by `RTK_BVH_LEAF_MAX_ITEMS` and probably _very_ small
		uint32_t i;
		for (i = 0; i != num_unique_meshes; i++) {
			if (unique_meshes[i] == mesh_ix) break;
		}
		if (i == num_unique_meshes) {
			unique_meshes[i] = mesh_ix;
			num_unique_meshes++;
		}
	}

	size_t num_tris_aligned = (num_tris + 3) & ~3u;

	// Aligned to 64 bytes
	size_t byte_size = sizeof(rtki_bvh_leaf);
	byte_size += sizeof(rtki_leaf_triangle) * num_tris_aligned;
	byte_size += sizeof(uint32_t) * num_unique_meshes;
	byte_size = rtki_align_up_sz(byte_size, 64u);
	rtki_atomic_size_add(&b->a_leaf_size, byte_size);
}

static void rtki_build_node_equal(rtk_task_ctx *ctx, rtki_build_node *node)
{
	rtk_build *b = ctx->build;

	rtki_reg3 size = rtki_sub3(rtki_load3x(&node->max), rtki_loadx3(&node->max));
	rtk_real max_comp = rtki_maxcomp3(size);
	int axis = rtki_x3(size) == max_comp ? 0 : rtki_y3(size) == max_comp ? 1 : 2;

	rtki_build_item *items = b->items + node->item_begin;
	qsort(items, node->item_count, sizeof(rtki_build_item), rtki_item_sort_axis_fn[axis]);

	size_t num_total = node->item_count;
	size_t num_left = num_total / 2;
	size_t num_right = num_total - num_left;

	size_t child_index = rtki_atomic_size_add(&b->a_nodes_allocated, 2);
	node->child_index = child_index;

	rtki_bounds3 bounds_left, bounds_right;
	rtki_bounds_reset(&bounds_left);
	rtki_bounds_reset(&bounds_right);

	// Calculate child bounds
	for (size_t i = 0; i < num_left; i++) {
		rtki_bounds3 bb = { rtki_load3x(&items[i].min), rtki_loadx3(&items[i].max) };
		rtki_bounds_add(&bounds_left, &bb);
	}
	for (size_t i = num_left; i < num_total; i++) {
		rtki_bounds3 bb = { rtki_load3x(&items[i].min), rtki_loadx3(&items[i].max) };
		rtki_bounds_add(&bounds_right, &bb);
	}

	rtki_build_node *child = b->nodes + child_index;
	child[0].child_index = SIZE_MAX;
	child[0].vertex_offset = SIZE_MAX;
	child[0].item_begin = node->item_begin;
	child[0].item_count = num_left;
	child[0].min = rtki_to_vec3(bounds_left.min);
	child[0].max = rtki_to_vec3(bounds_left.max);
	child[0].depth = node->depth + 1;
	child[1].child_index = SIZE_MAX;
	child[1].vertex_offset = SIZE_MAX;
	child[1].item_begin = node->item_begin + num_left;
	child[1].item_count = num_right;
	child[1].min = rtki_to_vec3(bounds_right.min);
	child[1].max = rtki_to_vec3(bounds_right.max);
	child[1].depth = node->depth + 1;

	double cost_left = (double)num_left * b->bvh_task_item_cost;
	double cost_right = (double)num_right * b->bvh_task_item_cost;
	rtki_push_task(ctx, &rtki_task_build_node, child_index, 0, cost_left);
	rtki_push_task(ctx, &rtki_task_build_node, child_index + 1, 0, cost_right);
}

static void rtki_build_node_sah(rtk_task_ctx *ctx, rtki_build_node *node)
{
	rtk_build *b = ctx->build;
	RTK_ALIGN16 rtki_build_bucket buckets[RTK_BUILD_SPLITS];

	rtk_real best_cost = RTK_INF;
	int best_axis = -1;
	int best_bucket;
	rtki_bounds3 best_bounds[2];

	rtki_bounds3 parent_bounds;
	parent_bounds.min = rtki_load3x(&node->min);
	parent_bounds.max = rtki_loadx3(&node->max);
	rtk_real rcp_parent_area = 1.0f / rtki_bounds_area(&parent_bounds);

	for (int axis = 0; axis < 3; axis++) {

		// Reset all buckets for the axis
		for (int i = 0; i < RTK_BUILD_SPLITS; i++) {
			rtki_bounds_reset(&buckets[i].bounds);
			buckets[i].num = 0;
		}

		rtk_real min = node->min.v[axis];
		rtk_real max = node->max.v[axis];
		rtk_real min_2x = min + min;
		rtk_real rcp_scale_2x = (0.5f * (rtk_real)RTK_BUILD_SPLITS) / (max - min);

		// Insert items into the buckets
		rtki_build_item* items = b->items + node->item_begin;
		for (size_t i = 0; i < node->item_count; i++) {
			rtki_build_item* item = &items[i];
			rtk_real mid_2x = item->min.v[axis] + item->max.v[axis];
			int bucket = (int)((mid_2x - min_2x) * rcp_scale_2x);
			if (bucket < 0) bucket = 0;
			if (bucket > RTK_BUILD_SPLITS - 1) bucket = RTK_BUILD_SPLITS - 1;
			rtki_build_bucket* b = &buckets[bucket];
			b->bounds.min = rtki_min3(b->bounds.min, rtki_load3x(&item->min));
			b->bounds.max = rtki_max3(b->bounds.max, rtki_loadx3(&item->max));
			b->num++;
		}

		// Scan backwards to get `bounds_right`
		buckets[RTK_BUILD_SPLITS - 1].bounds_right = buckets[RTK_BUILD_SPLITS - 1].bounds;
		for (int i = RTK_BUILD_SPLITS - 1; i > 0; i--) {
			rtki_build_bucket* b = &buckets[i];
			b[-1].bounds_right.min = rtki_min3(b[-1].bounds.min, b[0].bounds_right.min);
			b[-1].bounds_right.max = rtki_max3(b[-1].bounds.max, b[0].bounds_right.max);
		}

		// Scan forwards to find the best split
		rtki_bounds3 bounds_left;
		rtki_bounds_reset(&bounds_left);
		size_t num_left = 0;

		for (int i = 0; i < RTK_BUILD_SPLITS - 1; i++) {
			rtki_build_bucket* bucket = &buckets[i];
			rtki_build_bucket* bucket_right = &buckets[i + 1];
			rtki_bounds_add(&bounds_left, &bucket->bounds);

			num_left += bucket->num;
			size_t num_right = node->item_count - num_left;
			if (num_left == 0 || num_right == 0) continue;

			rtk_real area_left = rtki_bounds_area(&bounds_left);
			rtk_real area_right = rtki_bounds_area(&bucket_right->bounds_right);

			rtk_real cost_left = (rtk_real)((num_left + 3) / 4) * b->sah_item_cost;
			rtk_real cost_right = (rtk_real)((num_right + 3) / 4) * b->sah_item_cost;
			rtk_real split_cost = b->sah_split_cost + (area_left * cost_left + area_right * cost_right) * rcp_parent_area;

			if (split_cost < best_cost) {
				best_bounds[0] = bounds_left;
				best_bounds[1] = bucket_right->bounds_right;
				best_cost = split_cost;
				best_axis = axis;
				best_bucket = i;
			}
		}
	}

	rtk_real leaf_cost = (rtk_real)node->item_count * b->sah_item_cost;
	if (best_cost < leaf_cost || node->item_count > b->max_leaf_items) {

		// If we didn't find a good split axis just split the items equally in some direction
		if (best_axis < 0) {
			if (node->item_count > b->max_leaf_items) {
				rtki_build_node_leaf(ctx, node);
			} else {
				rtki_build_node_equal(ctx, node);
			}
			return;
		}

		// Split the node using the best axis and bucket, we need to replicate the
		// exact split criteria (bucket index) to make the heuristic accurate.
		rtk_real min = node->min.v[best_axis];
		rtk_real max = node->max.v[best_axis];
		rtk_real min_2x = min + min;
		rtk_real rcp_scale_2x = (0.5f * (rtk_real)RTK_BUILD_SPLITS) / (max - min);

		// Split the items
		rtki_build_item* items = b->items + node->item_begin;
		rtki_build_item* first = items;
		rtki_build_item* last = first + node->item_count;
		while (first != last) {
			rtk_real mid_2x = first->min.v[best_axis] + first->max.v[best_axis];
			int bucket = (int)((mid_2x - min_2x) * rcp_scale_2x);
			if (bucket < 0) bucket = 0;
			if (bucket > RTK_BUILD_SPLITS - 1) bucket = RTK_BUILD_SPLITS - 1;
			if (bucket <= best_bucket) {
				first++;
			}
			else {
				last--;
				rtki_build_item temp = *first;
				*first = *last;
				*last = temp;
			}
		}

		size_t num_left = first - items;
		size_t num_right = node->item_count - num_left;
		rtk_assert(num_left > 0 && num_right > 0);

		size_t child_index = rtki_atomic_size_add(&b->a_nodes_allocated, 2);
		node->child_index = child_index;

		rtki_build_node *child = b->nodes + child_index;
		child[0].child_index = SIZE_MAX;
		child[0].item_begin = node->item_begin;
		child[0].item_count = num_left;
		child[0].min = rtki_to_vec3(best_bounds[0].min);
		child[0].max = rtki_to_vec3(best_bounds[0].max);
		child[0].depth = node->depth + 1;
		child[1].child_index = SIZE_MAX;
		child[1].item_begin = node->item_begin + num_left;
		child[1].item_count = num_right;
		child[1].min = rtki_to_vec3(best_bounds[1].min);
		child[1].max = rtki_to_vec3(best_bounds[1].max);
		child[1].depth = node->depth + 1;

		double cost_left = (double)num_left * b->bvh_task_item_cost;
		double cost_right = (double)num_right * b->bvh_task_item_cost;
		rtki_push_task(ctx, &rtki_task_build_node, child_index, 0, cost_left);
		rtki_push_task(ctx, &rtki_task_build_node, child_index + 1, 0, cost_right);

		rtki_atomic_size_add(&b->a_depth_num_nodes[node->depth], 1);

	} else {
		rtki_build_node_leaf(ctx, node);
	}
}

static void rtki_task_start(const rtk_task *task, rtk_task_ctx *ctx);
static void rtki_task_setup_triangles(const rtk_task *task, rtk_task_ctx *ctx);
static void rtki_task_start_build_nodes(const rtk_task *task, rtk_task_ctx *ctx);
static void rtki_task_build_node(const rtk_task *task, rtk_task_ctx *ctx);
static void rtki_task_start_finalize_nodes(const rtk_task *task, rtk_task_ctx *ctx);
static void rtki_task_finalize_node(const rtk_task *task, rtk_task_ctx *ctx);

static void rtki_decode_indices(const rtk_mesh *mesh, uint32_t *dst, size_t offset, size_t count)
{
	if (mesh->index_cb) {
		mesh->index_cb(mesh->index_cb_user, mesh, dst, offset, count);
		return;
	}

	if (mesh->index.data) {
		if (mesh->index.type == RTK_TYPE_U16) {
			size_t stride = mesh->index.stride ? mesh->index.stride : sizeof(uint16_t) * 3;
			const void *data = (const char*)mesh->index.data + offset * stride;
			const uint16_t *ix = (const uint16_t*)data;
			for (size_t i = 0; i < count; i++) {
				dst[0] = ix[0];
				dst[1] = ix[1];
				dst[2] = ix[2];
				ix += 3;
				dst += 3;
			}
		} else if (mesh->index.type == RTK_TYPE_U32) {
			size_t stride = mesh->index.stride ? mesh->index.stride : sizeof(uint32_t) * 3;
			const void *data = (const char*)mesh->index.data + offset * stride;
			const uint32_t *ix = (const uint32_t*)data;
			for (size_t i = 0; i < count; i++) {
				dst[0] = ix[0];
				dst[1] = ix[1];
				dst[2] = ix[2];
				ix += 3;
				dst += 3;
			}
		} else {
			rtk_assert(0 && "Bad index type");
		}
	} else {
		for (size_t i = 0; i < count; i++) {
			uint32_t base = (uint32_t)(offset + i) * 3;
			dst[0] = base + 0;
			dst[1] = base + 1;
			dst[2] = base + 2;
				dst += 3;
		}
	}
}

static void rtki_decode_vertices(const rtk_mesh *mesh, rtk_vec3 *dst, const uint32_t *indices, size_t count)
{
	if (mesh->position_cb) {
		mesh->position_cb(mesh->position_cb_user, mesh, dst, indices, count);
		return;
	}

	rtk_assert(mesh->position.data);
	rtk_type type = mesh->position.type;
	if (type == RTK_TYPE_REAL) {
		type = sizeof(rtk_real) == sizeof(double) ? RTK_TYPE_F64 : RTK_TYPE_F32;
	}

	if (type == RTK_TYPE_F32) {
		size_t stride = mesh->position.stride ? mesh->position.stride : sizeof(float) * 3;
		const char *data = (const char*)mesh->position.data;
		for (size_t i = 0; i < count; i++) {
			size_t base = i * 3;
			const float *a = (const float*)(data + indices[base + 0] * stride);
			const float *b = (const float*)(data + indices[base + 1] * stride);
			const float *c = (const float*)(data + indices[base + 2] * stride);
			dst[0].x = (rtk_real)a[0]; dst[0].y = (rtk_real)a[1]; dst[0].z = (rtk_real)a[2];
			dst[1].x = (rtk_real)b[0]; dst[1].y = (rtk_real)b[1]; dst[1].z = (rtk_real)b[2];
			dst[2].x = (rtk_real)c[0]; dst[2].y = (rtk_real)c[1]; dst[2].z = (rtk_real)c[2];
			dst += 3;
		}
	} else if (type == RTK_TYPE_F64) {
		size_t stride = mesh->position.stride ? mesh->position.stride : sizeof(double) * 3;
		const char *data = (const char*)mesh->position.data;
		for (size_t i = 0; i < count; i++) {
			size_t base = i * 3;
			const float *a = (const float*)(data + indices[base + 0] * stride);
			const float *b = (const float*)(data + indices[base + 1] * stride);
			const float *c = (const float*)(data + indices[base + 2] * stride);
			dst[0].x = (rtk_real)a[0]; dst[0].y = (rtk_real)a[1]; dst[0].z = (rtk_real)a[2];
			dst[1].x = (rtk_real)b[0]; dst[1].y = (rtk_real)b[1]; dst[1].z = (rtk_real)b[2];
			dst[2].x = (rtk_real)c[0]; dst[2].y = (rtk_real)c[1]; dst[2].z = (rtk_real)c[2];
			dst += 3;
		}
	} else {
		rtk_assert(0 && "Bad index type");
	}
}

static void rtki_task_setup_triangles(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;

	rtki_triangle_task_state *state = &b->task.triangle.state[task->index];

	size_t offset = state->base;
	size_t left = state->num;
	rtki_build_log(b, "Gathering triangles [%zu, %zu] of %zu", offset, offset + left, b->num_triangles);

	rtki_bounds3 bounds;
	rtki_bounds_reset(&bounds);

	rtki_build_item *item = b->items + offset;

	for (size_t mesh_ix = 0; mesh_ix < b->desc.num_meshes && left > 0; mesh_ix++) {
		const rtk_mesh *mesh = &b->desc.meshes[mesh_ix];
		if (offset >= mesh->num_triangles) {
			offset -= mesh->num_triangles;
			continue;
		}

		size_t mesh_left = mesh->num_triangles - offset;
		if (mesh_left > left) mesh_left = left;

		while (mesh_left > 0) {
			size_t chunk_size = mesh_left;
			if (chunk_size > 128) chunk_size = 128;
			uint32_t indices[128*3];
			rtk_vec3 vertices[128*3 + 1];

			rtki_decode_indices(mesh, indices, offset, chunk_size);
			rtki_decode_vertices(mesh, vertices, indices, chunk_size);

			for (size_t i = 0; i < chunk_size; i++) {
				size_t base = i * 3;
				rtki_reg3 a = rtki_load3x(&vertices[base + 0]);
				rtki_reg3 b = rtki_load3x(&vertices[base + 1]);
				rtki_reg3 c = rtki_load3x(&vertices[base + 2]);
				rtki_bounds3 bb = { a, a };
				rtki_bounds_add_point(&bb, b);
				rtki_bounds_add_point(&bb, c);
				rtki_bounds_add(&bounds, &bb);

				item->min = rtki_to_vec3(bb.min);
				item->max = rtki_to_vec3(bb.max);
				item->v[0].position = vertices[base + 0];
				item->v[0].index = indices[base + 0];
				item->v[1].position = vertices[base + 1];
				item->v[1].index = indices[base + 1];
				item->v[2].position = vertices[base + 2];
				item->v[2].index = indices[base + 2];
				item->mesh_index = (uint32_t)mesh_ix;
				item->triangle_index = (uint32_t)(offset + i);
				item++;
			}

			offset += chunk_size;
			mesh_left -= chunk_size;
		}

		offset = 0;
	}

	state->min = rtki_to_vec3(bounds.min);
	state->max = rtki_to_vec3(bounds.max);
}

// Vertex sets

#define RTK_VERTEX_SET_MAX_SIZE 256

// Sorted array of vertex entries
// [0:32] uint32_t vertex_index;
// [32:64] uint32_t mesh_index;
typedef struct {
	uint64_t entries[RTK_VERTEX_SET_MAX_SIZE];
	size_t size;
} rtki_vertex_set;

static void rtki_vertex_set_insert(rtki_vertex_set *set, uint32_t mesh_ix, uint32_t vertex_ix)
{
	uint64_t entry = (uint64_t)mesh_ix << 32u | vertex_ix;
	size_t size = set->size;
	for (size_t i = 0; i < size; i++) {
		if (set->entries[i] >= entry) {
			if (set->entries[i] == entry) return; // Found existing!
			// Found a slot to insert to, push the rest back
			for (; i < size; i++) {
				uint64_t tmp = set->entries[i];
				set->entries[i] = entry;
				entry = tmp;
			}
			break;
		}
	}
	set->entries[size] = entry;
	set->size = size + 1;
}

static bool rtki_vertex_set_merge(rtki_vertex_set *dst, const rtki_vertex_set *a, const rtki_vertex_set *b)
{
	const uint64_t *ap = a->entries, *aend = ap + a->size;
	const uint64_t *bp = b->entries, *bend = bp + b->size;
	uint64_t *dp = dst->entries, *dend = dp + RTK_VERTEX_SET_MAX_SIZE;

	// Branchlessly merge A and B in the correct order
	while (ap != aend && bp != bend) {
		if (dp == dend) return false;
		uint64_t av = *ap, bv = *bp;
		int ax = av <= bv ? 1 : 0;
		int bx = bv <= av ? 1 : 0;
		ap += ax;
		bp += bx;
		*dp++ = ax ? av : bv;
	}

	// Copy the rest from A and B
	while (ap != aend) {
		if (dp == dend) return false;
		*dp++ = *ap++;
	}
	while (bp != bend) {
		if (dp == dend) return false;
		*dp++ = *bp++;
	}

	dst->size = dp - dst->entries;
	return true;
}

static void rtki_vertex_set_copy(rtki_vertex_set *dst, const rtki_vertex_set *src)
{
	dst->size = src->size;
	memcpy(dst->entries, src->entries, src->size * sizeof(uint64_t));
}

// Find `entry` in the set, returns ~0u if not found
static uint32_t rtki_vertex_set_find(const rtki_vertex_set *set, uint32_t mesh_ix, uint32_t vertex_ix)
{
	uint64_t entry = (uint64_t)mesh_ix << 32u | vertex_ix;
	const uint64_t *lo = set->entries;
	const uint64_t *hi = lo + set->size;

	// Binary search down to a range
	while (hi - lo > 8) {
		size_t dist = hi - lo;
		const uint64_t *mid = lo + (dist >> 1);
		if (*mid > entry) {
			hi = mid;
		} else {
			lo = mid;
		}
	}

	// Linear search for the rest
	for (; lo != hi; lo++) {
		if (*lo == entry) {
			return (uint32_t)(lo - set->entries);
		}
	}

	rtk_assert(0 && "Vertex not found in set");
	return ~0u;
}

static void rtki_build_assign_vertices(rtk_build *b, rtki_build_node *node, const rtki_vertex_set *set, size_t vertex_offset)
{
	// Return if this node has a vertex group already
	if (node->vertex_offset != SIZE_MAX) return;
	node->vertex_offset = vertex_offset;
	if (node->child_index != ~0u) {
		rtki_build_node *child = b->nodes + node->child_index;
		rtki_build_assign_vertices(b, &child[0], set, vertex_offset);
		rtki_build_assign_vertices(b, &child[1], set, vertex_offset);
	} else {
		rtki_build_item *items = b->items + node->item_begin;
		for (size_t i = 0; i < node->item_count; i++) {
			rtki_build_item *item = &items[i];
			if (item->mesh_index == ~0u) continue;
			item->v_ix[0] = (uint8_t)rtki_vertex_set_find(set, item->mesh_index, item->v[0].index);
			item->v_ix[1] = (uint8_t)rtki_vertex_set_find(set, item->mesh_index, item->v[1].index);
			item->v_ix[2] = (uint8_t)rtki_vertex_set_find(set, item->mesh_index, item->v[2].index);
		}
	}
}

// Close the vertex set `set` copying the data into the contiguous array and
// assigning vertices in related nodes.
static void rtki_build_close_vertices(rtk_build *b, rtki_build_node *node, const rtki_vertex_set *set)
{
	size_t vertex_offset = rtki_atomic_size_add(&b->a_vertex_count, set->size);
	rtki_build_assign_vertices(b, node, set, vertex_offset);
}

// Gather vertices from `node_ptr` into `parent_set`. Returns true if
// the vertex group is still open ie. it can be appended into.
static bool rtki_build_gather_vertices(rtk_build *b, rtki_build_node *node, rtki_vertex_set *parent_set)
{
	if (node->child_index != SIZE_MAX) {
		rtki_build_node *child = b->nodes + node->child_index;

		rtki_vertex_set set[2];
		set[0].size = set[1].size = 0;
		int open[2];
		open[0] = rtki_build_gather_vertices(b, &child[0], &set[0]);
		open[1] = rtki_build_gather_vertices(b, &child[1], &set[1]);

		// If both sets are closed we're done with this node
		if (!open[0] && !open[1]) return false;

		// If both sets are open try to merge them
		if (open[0] && open[1]) {
			if (rtki_vertex_set_merge(parent_set, &set[0], &set[1])) {
				return true;
			}

			// Close the larger child and pass along the smaller one
			int close_ix = set[1].size > set[0].size ? 1 : 0;
			rtki_build_close_vertices(b, &child[close_ix], &set[close_ix]);
			rtki_vertex_set_copy(parent_set, &set[close_ix ^ 1]);
			return 1;
		}

		// Pass along the currently open set
		int copy_ix = open[1] ? 1 : 0;
		rtki_vertex_set_copy(parent_set, &set[copy_ix]);
		return 1;

	} else {

		// Insert vertices of all the triangles in this node
		rtki_build_item *items = b->items + node->item_begin;
		for (size_t i = 0; i < node->item_count; i++) {
			rtki_build_item *item = &items[i];
			uint32_t mesh_ix = item->mesh_index;
			if (mesh_ix == ~0u) continue;
			rtki_vertex_set_insert(parent_set, mesh_ix, item->v[0].index);
			rtki_vertex_set_insert(parent_set, mesh_ix, item->v[1].index);
			rtki_vertex_set_insert(parent_set, mesh_ix, item->v[2].index);
		}

		return 1;
	}
}

static void rtki_task_start(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;
	rtki_build_log(b, "Starting build");

	size_t num_tri_tasks = RTK_MAX_CONCURRENT_TASKS;
	if (num_tri_tasks > ctx->queue_capacity) num_tri_tasks = ctx->queue_capacity;

	const size_t min_task_tris = 1024; 
	size_t task_tris = b->num_triangles;
	if (num_tri_tasks > 0 && b->num_triangles > min_task_tris) {
		task_tris = (task_tris + num_tri_tasks - 1) / num_tri_tasks;
		if (task_tris < min_task_tris) task_tris = min_task_tris;
	}

	b->task.triangle.count = (b->num_triangles + task_tris - 1) / task_tris;

	size_t task_i = 0;
	for (size_t base = 0; base < b->num_triangles; base += task_tris) {
		size_t num = b->num_triangles - base;
		if (num > task_tris) num = task_tris;
		double cost = (double)num * b->triangle_task_item_cost;
		b->task.triangle.state[task_i].base = base;
		b->task.triangle.state[task_i].num = num;
		rtki_push_task(ctx, &rtki_task_setup_triangles, task_i, 0, cost);
		task_i++;
	}

	b->next_task.fn = &rtki_task_start_build_nodes;
}

static void rtki_task_start_build_nodes(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;
	rtki_build_log(b, "Starting to build nodes");

	rtki_bounds3 bounds;
	rtki_bounds_reset(&bounds);
	for (size_t i = 0; i < b->task.triangle.count; i++) {
		rtki_triangle_task_state *state = &b->task.triangle.state[i];
		rtki_bounds3 bb = { rtki_load3x(&state->min), rtki_loadx3(&state->max) };
		rtki_bounds_add(&bounds, &bb);
	}

	b->a_nodes_allocated = 1;
	b->nodes[0].child_index = SIZE_MAX;
	b->nodes[0].vertex_offset = SIZE_MAX;
	b->nodes[0].item_begin = 0;
	b->nodes[0].item_count = b->num_triangles;
	b->nodes[0].min = rtki_to_vec3(bounds.min);
	b->nodes[0].max = rtki_to_vec3(bounds.max);
	b->nodes[0].depth = 0;

	double cost = b->nodes[0].item_count * b->bvh_task_item_cost;
	rtki_push_task(ctx, &rtki_task_build_node, 0, 0, cost);

	b->next_task.fn = &rtki_task_start_finalize_nodes;
}

static void rtki_task_build_node(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;
	rtki_build_node *node = &b->nodes[task->index];

	rtki_build_log(b, "Building node %zu (%zu items)", task->index, node->item_count);

	// Reached the end, must do a leaf.
	if (node->depth == RTK_BVH_MAX_DEPTH) {
		rtki_build_node_leaf(ctx, node);
		return;
	}

	// Calculate the amount of triangles/primitives per node if we _don't_
	// start splitting them evenly at this level. If these exceed the limits
	// then we must split evenly.
	uint64_t splits_left = RTK_BVH_MAX_DEPTH - node->depth - 1;
	if (splits_left > 63) splits_left = 63;
	uint64_t split_items = node->item_count >> splits_left;
	if (split_items > b->max_leaf_items) {
		rtki_build_node_equal(ctx, node);
		return;
	}

	// Create a leaf node if there's not many triangles or primitives
	if (node->item_count <= RTK_BVH_LEAF_MIN_ITEMS) {
		rtki_build_node_leaf(ctx, node);
		return;
	}

	// Default: Use SAH
	rtki_build_node_sah(ctx, node);
}

static void rtki_task_start_finalize_nodes(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;
	rtki_build_log(b, "Starting to finalize nodes");

	// If the root node is a leaf generate a virtual root
	rtki_build_node *root = &b->nodes[0];
	if (root->child_index == SIZE_MAX) {
		rtki_build_node *left = &b->nodes[2];
		rtki_build_node *right = &b->nodes[2];
		*left = *root;
		left->depth = 1;
		right->min = right->max = root->min;
		right->item_begin = right->item_count = 0;
		right->child_index = SIZE_MAX;
		right->vertex_offset = SIZE_MAX;
		right->depth = 1;

		rtki_atomic_size_add(&b->a_depth_num_nodes[0], 1);

		root->child_index = 1;
	}

	double cost = b->nodes[0].item_count * b->finalize_task_item_cost;
	rtki_push_task(ctx, &rtki_task_finalize_node, 0, 0, cost);

	b->next_task.fn = NULL;
}

static void rtki_task_finalize_node(const rtk_task *task, rtk_task_ctx *ctx)
{
	rtk_build *b = task->build;
	rtki_build_node *node = &b->nodes[task->index];

	rtki_build_log(b, "Finalizing node %zu (%zu items)", task->index, node->item_count);

	if (node->item_count >= 2048 && node->child_index != SIZE_MAX) {

		rtki_build_node *child = &b->nodes[node->child_index];
		double cost_left = child[0].item_count * b->finalize_task_item_cost;
		double cost_right = child[1].item_count * b->finalize_task_item_cost;
		rtki_push_task(ctx, &rtki_task_finalize_node, node->child_index, 0, cost_left);
		rtki_push_task(ctx, &rtki_task_finalize_node, node->child_index + 1, 0, cost_right);

	} else {

		rtki_vertex_set root_set;
		root_set.size = 0;
		if (rtki_build_gather_vertices(b, node, &root_set)) {
			rtki_build_close_vertices(b, node, &root_set);
		}
	}
}

static void rtki_linearize_leaf(rtk_build *b, rtki_bvh_leaf *dst, rtki_build_node *src)
{
	char *begin = (char*)dst;

	uint32_t num_tris = 0;
	rtki_build_item *items = b->items + src->item_begin;
	for (size_t i = 0; i < src->item_count; i++) {
		rtki_build_item *item = &items[i];
		uint32_t mesh_ix = item->mesh_index;
		if (mesh_ix == ~0u) continue;
		num_tris++;

		for (size_t vi = 0; vi < 3; vi++) {
			size_t vertex_ix = src->vertex_offset + item->v_ix[vi];
			uint32_t mask_bit = 1u << (uint32_t)(vertex_ix & 31u);
			size_t mask_ix = vertex_ix >> 5;
			if ((b->vertex_written_mask[mask_ix] & mask_bit) == 0) {
				b->vertex_written_mask[mask_ix] |= mask_bit;
				b->final_vertices[vertex_ix] = item->v[vi];
			}
		}
	}
	size_t num_tris_aligned = (num_tris + 3) & ~3u;

	uint64_t triangle_info = num_tris | (b->vertex_data_offset + src->vertex_offset);
	rtki_leaf_triangle *tris_begin = (rtki_leaf_triangle*)(dst + 1);
	rtki_leaf_triangle *tris_end = (rtki_leaf_triangle*)(tris_begin + num_tris_aligned);
	uint32_t *unique_meshes = (uint32_t*)tris_end;
	char *end = (char*)unique_meshes;
	size_t size = end - begin;
	b->final_leaf_offset += size;
	b->final_leaf_offset = rtki_align_up_sz(b->final_leaf_offset, 64);

	rtki_leaf_triangle *tri = tris_begin;

	uint32_t num_unique_meshes = 0;
	for (size_t i = 0; i < src->item_count; i++) {
		rtki_build_item *item = &items[i];
		uint32_t mesh_ix = item->mesh_index;
		if (mesh_ix == ~0u) continue;
		num_tris++;

		tri->v[0] = item->v_ix[0];
		tri->v[1] = item->v_ix[1];
		tri->v[2] = item->v_ix[2];
		tri->triangle_index = item->triangle_index;

		// O(n^2) but n is bounded by `RTK_BVH_LEAF_MAX_ITEMS` and probably _very_ small
		uint32_t i;
		for (i = 0; i != num_unique_meshes; i++) {
			if (unique_meshes[i] == mesh_ix) {
				tri->local_mesh_index = (uint8_t)i;
			}
		}
		if (i == num_unique_meshes) {
			unique_meshes[i] = mesh_ix;
			num_unique_meshes++;
		}
	}
}

static void rtki_linearize_node(rtk_build *b, rtki_bvh_node *dst, rtki_build_node *src)
{
	for (unsigned i = 0; i < 4; i++) {
		uint32_t mid_i = i >> 1;
		uint32_t child_i = i & 1;

		// Choose the left/right child of a middle child or middle/NULL if it's a leaf
		rtki_build_node *mid_src = &b->nodes[src->child_index + mid_i];
		rtki_build_node *child_src;
		if (mid_src->child_index != SIZE_MAX) {
			child_src = &b->nodes[mid_src->child_index + child_i];
		} else {
			if (child_i == 0) {
				child_src = mid_src;
			} else {
				child_src = NULL;
			}
		}

		// Replace empty nodes with NULL leaves
		if (child_src != NULL && child_src->item_count == 0) {
			child_src = NULL;
		}

		if (child_src) {
			dst->bounds_x[0][i] = child_src->min.x;
			dst->bounds_x[1][i] = child_src->max.x;
			dst->bounds_y[0][i] = child_src->min.y;
			dst->bounds_y[1][i] = child_src->max.y;
			dst->bounds_z[0][i] = child_src->min.z;
			dst->bounds_z[1][i] = child_src->max.z;

			if (child_src->child_index != SIZE_MAX) {
				size_t offset = b->depth_node_offset[src->depth + 2]++;
				rtki_bvh_node *child_dst = b->final_nodes + offset;
				dst->ptr[i] = b->node_data_offset + offset;
				rtki_linearize_node(b, child_dst, child_src);
			} else {
				rtki_bvh_leaf *leaf_dst = (rtki_bvh_leaf*)((char*)b->final_leaves + b->final_leaf_offset);
				dst->ptr[i] = (b->leaf_data_offset + b->final_leaf_offset) | 1;
				rtki_linearize_leaf(b, leaf_dst, child_src);
			}
		} else {
			dst->bounds_x[0][i] = +1.0f;
			dst->bounds_x[1][i] = -1.0f;
			dst->bounds_y[0][i] = +1.0f;
			dst->bounds_y[1][i] = -1.0f;
			dst->bounds_z[0][i] = +1.0f;
			dst->bounds_z[1][i] = -1.0f;
			dst->ptr[i] = b->leaf_data_offset;
		}
	}
}


rtk_build *rtk_start_build(const rtk_scene_desc *desc, rtk_task *first_task)
{
	size_t num_items = 0;
	size_t num_triangles = 0;
	size_t num_nodes = 0;

	for (size_t i = 0; i < desc->num_meshes; i++) {
		const rtk_mesh *mesh = &desc->meshes[i];
		num_items += mesh->num_triangles;
		num_triangles += mesh->num_triangles;
	}

	// TODO: Better lower bound
	num_nodes = num_items;
	if (num_nodes < 4) num_nodes = 4;

	size_t num_vertex_masks = (num_triangles * 3 + 31u) / 32u;

	size_t size = sizeof(rtk_build);
	size += sizeof(rtki_build_item) * num_items;
	size += sizeof(rtki_build_node) * num_nodes;
	size += sizeof(uint32_t) * num_vertex_masks;
	void *data = rtk_mem_alloc(size);
	if (!data) return NULL;
	void *data_end = (char*)data + size;

	rtk_build *build = (rtk_build*)data;
	memset(build, 0, sizeof(rtk_build));

	build->items = (rtki_build_item*)(build + 1);
	build->nodes = (rtki_build_node*)(build->items + num_items);
	build->vertex_written_mask = (uint32_t*)(build->nodes + num_nodes);
	rtk_assert(build->vertex_written_mask + num_vertex_masks == data_end);
	build->num_vertex_masks = num_vertex_masks;

	build->alloc_size = size;
	build->desc = *desc;
	build->num_triangles = num_triangles;

	build->min_task_cost = 1000.0;
	build->bvh_task_item_cost = 10.0;
	build->finalize_task_item_cost = 5.0;
	build->triangle_task_item_cost = 10.0;
	build->next_task.build = build;

	build->max_leaf_items = RTK_BVH_LEAF_MAX_ITEMS;

	build->a_tasks_left = 1;

	build->desc = *desc;

	// Reserve space for the NULL leaf
	build->a_leaf_size = rtki_align_up_sz(sizeof(rtki_bvh_leaf), 64);

	if (first_task) {
		first_task->build = build;
		first_task->fn = &rtki_task_start;
	} else {
		// TODO
		rtk_task task;
		task.build = build;
		task.fn = &rtki_task_start;
		rtk_run_task(&task, NULL, 0);
	}
	return build;
}

size_t rtk_run_task(const rtk_task *task, rtk_task *queue, size_t queue_size)
{
	rtk_build *b = task->build;

	rtk_task_ctx ctx;
	ctx.build = task->build;
	ctx.queue = queue;
	ctx.queue_capacity = queue_size;
	ctx.queue_num = 0;
	task->fn(task, &ctx);

	size_t num_left = rtki_atomic_size_add(&b->a_tasks_left, -1) - 1;
	while (num_left == 0) {
		if (b->next_task.fn) {
			rtki_atomic_size_add(&b->a_tasks_left, 1);
			rtk_task next_task = b->next_task;
			next_task.fn(&next_task, &ctx);
			num_left = rtki_atomic_size_add(&b->a_tasks_left, -1) - 1;
		} else {
			rtk_assert(ctx.queue_num == 0);
			break;
		}
	}

	return ctx.queue_num;
}

size_t rtk_get_build_size(const rtk_build *build)
{
	size_t size = sizeof(rtk_scene);
	size = rtki_align_up_sz(size, 128);
	size += build->a_nodes_allocated * sizeof(rtki_bvh_node);
	size = rtki_align_up_sz(size, 128);
	size += build->a_leaf_size;
	size = rtki_align_up_sz(size, 128);
	size += build->a_vertex_count * sizeof(rtk_vertex);
	size = rtki_align_up_sz(size, 128);
	return size;
}

rtk_scene *rtk_finish_build_to(rtk_build *build, void *buffer, size_t size)
{
	size_t required_size = rtk_get_build_size(build);
	if (size < required_size) return NULL;
	rtk_scene *scene = (rtk_scene*)buffer;
	memcpy(scene->magic, "\x00RTK\r\n\x1a\x0a", 8);
	scene->endian = 0xaabb;
	scene->sizeof_real = sizeof(rtk_real);
	scene->pad_0 = 0;
	scene->version = 1;
	scene->pad_1 = 0;
	scene->size_in_bytes = required_size;

	size_t offset = sizeof(rtk_scene);
	offset = rtki_align_up_sz(offset, 128);
	scene->node_offset = offset;
	offset += build->a_nodes_allocated * sizeof(rtki_bvh_node);
	offset = rtki_align_up_sz(offset, 128);
	scene->leaf_offset = offset;
	offset += build->a_leaf_size;
	offset = rtki_align_up_sz(offset, 128);
	scene->vertex_offset = offset;
	offset += build->a_vertex_count * sizeof(rtk_vertex);
	offset = rtki_align_up_sz(offset, 128);
	rtk_assert(offset == required_size);

	char *data = (char*)scene;
	build->final_nodes = (rtki_bvh_node*)(data + (size_t)scene->node_offset);
	build->final_leaves = (rtki_bvh_leaf*)(data + (size_t)scene->leaf_offset);
	build->final_vertices = (rtk_vertex*)(data + (size_t)scene->vertex_offset);

	rtki_bvh_leaf *null_leaf = build->final_leaves;
	null_leaf->triangle_info = 0;
	build->leaf_data_offset = rtki_align_up_sz(sizeof(rtki_bvh_leaf), 64);

	memset(build->vertex_written_mask, 0, sizeof(uint32_t) * build->num_vertex_masks);

	rtki_linearize_node(build, build->final_nodes, &build->nodes[0]);

	rtk_mem_free(build, build->alloc_size);

	return scene;
}

rtk_scene *rtk_finish_build(rtk_build *build)
{
	size_t size = rtk_get_build_size(build);
	void *buffer = rtk_mem_alloc(size);
	if (!buffer) {
		rtk_mem_free(build, build->alloc_size);
		return NULL;
	}

	return rtk_finish_build_to(build, buffer, size);
}

rtk_scene *rtk_build_scene(const rtk_scene_desc *desc)
{
	rtk_build *build = rtk_start_build(desc, NULL);
	return rtk_finish_build(build);
}

void rtk_free_scene(rtk_scene *scene)
{
	rtk_mem_free(scene, (size_t)scene->size_in_bytes);
}

#endif
