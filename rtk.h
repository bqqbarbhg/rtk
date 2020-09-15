#ifndef RTK_HEADER_INCLUDED
#define RTK_HEADER_INCLUDED

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// -- Constants

#define RTK_SCENE_VERSION 1

// -- General types

#define RTK_USER_DATA_SIZE 16

#define RTK_BUILD_DONE SIZE_MAX

#if defined(RTK_USE_DOUBLE) && RTK_USE_DOUBLE
	typedef double rtk_real;
	#define RTK_INF (1.79769e+308)
#else
	typedef float rtk_real;
	#define RTK_INF (3.40282e+38f)
#endif

typedef struct rtk_vec3 {
	union {
		struct { rtk_real x, y, z; };
		rtk_real v[3];
	};
} rtk_vec3;

typedef struct rtk_bounds {
	rtk_vec3 min, max;
} rtk_bounds;

typedef struct rtk_vertex {
	rtk_vec3 position;
	uint32_t index;
} rtk_vertex;

typedef union rtk_user_data {
	uint64_t u64[RTK_USER_DATA_SIZE / 8];
	unsigned char u8[RTK_USER_DATA_SIZE];
} rtk_user_data;

// -- Ray tracing types

typedef struct rtk_ray {
	rtk_vec3 origin;    // < Starting point of the ray
	rtk_vec3 direction; // < Direction of positive t, no need to be normalized
	rtk_real min_t;     // < Minimum t value eg, 0.0 for normal rays
	rtk_real max_t;     // < Maximum t value, use RTK_INF for unbounded
} rtk_ray;

typedef struct rtk_hit {
	rtk_real t;               // < Ray parameter t, distance if the ray is normalized
	rtk_real u, v;            // < Triangle barycentric coordinates / primitive parametrization
	rtk_vertex vertex[3];     // < Hit triangle vertices
	uint32_t mesh_index;      // < Index of the mesh in the scene (~0u if hit primitive)
	uint32_t triangle_index;  // < Triangle index in the mesh (~0u if hit primitive)
	uint32_t primitive_index; // < Index of the primitive in the scene (~0u if hit mesh)
	rtk_user_data user_data;  // < Arbitrary user data
} rtk_hit;

// -- Scene representation

// Absolute file offset in bytes
typedef uint64_t rtk_offset;

// BVH internal (4-way) node
typedef struct rtk_bvh_node {
	// Minimum/maximum bounds per child in structure-of-arrays layout
	rtk_real bounds_x[2][4];
	rtk_real bounds_y[2][4];
	rtk_real bounds_z[2][4];

	// Tagged child offset to `rtk_bvh_node` (LSB=0) or `rtk_leaf` (LSB=1)
	// The leaf tag LSB is not a part of the offset and should be cleared.
	rtk_offset tagged_offsets[4];
} rtk_bvh_node; 

// BVH leaf node: The leaf data is packed after the node header as follows:
//   rtk_leaf leaf;
//   rtk_leaf_triangle triangles[leaf.num_triangles];
//   rtk_leaf_primitive primitives[leaf.num_primitives];
//   rtk_leaf_meshes meshes[leaf.num_meshes];
//   char padding[]; // Padded to 64 bytes, see `leaf.size_in_bytes`
typedef struct rtk_leaf {
	// Offset to `rtk_vertex` array that triangles refer to
	rtk_offset vertex_data;

	// Size of all the data in this leaf in memory including the size of
	// this `rtk_leaf` structure
	uint32_t size_in_bytes;

	// Number of trailing elements, see comment above for the layout
	uint8_t num_triangles;
	uint8_t num_primitives;
	uint8_t num_meshes;

	uint8_t pad_0[1];
} rtk_leaf;

typedef struct rtk_leaf_triangle {
	uint8_t leaf_vertices[3]; // < Vertex indices into the `leaf.vertex_data` array
	uint8_t leaf_mesh;        // < Index into the leaf implicit `meshes` array
	uint32_t triangle_index;  // < Index of the triangle in the user-provided `rtk_mesh_desc`
} rtk_leaf_triangle; 

typedef struct rtk_leaf_mesh {
	uint32_t mesh_index; // < Index of the mesh in the user-provided `rtk_scene_desc`
	rtk_user_data user_data; // < Arbitrary user data from `rtk_mesh_desc`
} rtk_leaf_mesh; 

typedef struct rtk_leaf_primitive {
	rtk_bounds bounds;       // < Axis-aligned bounding box
	uint32_t prim_index;     // < Index of the primitive in `rtk_scene_desc`
	rtk_user_data user_data; // < Arbitrary user data from `rtk_primitive_desc`
} rtk_leaf_primitive; 

typedef struct rtk_scene_header {
	char magic[8];           // < "\0RTK\r\n\x1a\n"
	uint16_t endian;         // < `0xaabb` in the native endian
	uint8_t sizeof_real;     // < `sizeof(rtk_real)`
	uint8_t sizeof_user;     // < `sizeof(rtk_user_data)`
	uint32_t version;        // < `RTK_SCENE_VERSION`
	uint32_t pad_1;
	uint64_t size_in_bytes;  // < Size of the whole scene in bytes
} rtk_scene_header;

// Scene root. The scene data is a single pointer-free contiguous allocation
// so feel free to serialize it. The serialized scene is not portable between different
// endian systems or compile-time RTK configurations eg. `RTK_USE_DOUBLE`.
typedef struct rtk_scene {
	rtk_scene_header header; // < Header for checking file integrity
	uint64_t node_offset;    // < Byte offset of the first `rtk_bvh_node` (always 128)
	uint64_t leaf_offset;    // < Byte offset of the first `rtk_leaf`
	uint64_t vertex_offset;  // < Byte offset of the first `rtk_vertex`
} rtk_scene;

// -- Scene building

typedef enum rtk_datatype {
	RTK_DATATYPE_DEFAULT, // < Default type depends on usage
	RTK_DATATYPE_REAL,    // < `rtk_real`
	RTK_DATATYPE_F32,     // < `float`
	RTK_DATATYPE_F64,     // < `double`
	RTK_DATATYPE_U16,     // < `uint16_t`
	RTK_DATATYPE_U32,     // < `uint32_t`
	RTK_DATATYPE_U64,     // < `uint64_t`
	RTK_DATATYPE_SIZE_T,  // < `size_t`
} rtk_datatype;

typedef struct rtk_buffer_desc {
	const void *data;  // < Base data pointer
	size_t stride;     // < Number of bytes between elements
	rtk_datatype type; // < Data type of the elements (or default)
} rtk_buffer_desc;

// Fetch `triangle_count` triangles of vertices indexed by `indices` to `dst`, eg.
//   for (size_t i = 0; i < triangle_count*3; i++) { dst[i] = my_vertex_data[indices[i]]; }
typedef void rtk_vertex_func(void *user, rtk_vec3 *dst, const uint32_t *indices, size_t triangle_count);

typedef struct rtk_vertex_desc {
	rtk_buffer_desc buffer;  // < Default type: RTK_TYPE_REAL
	rtk_vertex_func *func; // < Custom callback to fetch vertices (overrides `buffer`)
	void *func_user;         // < User context pointer passed to `func`
} rtk_vertex_desc;

// Fetch `triangle_count` indices starting from `triangle_offset` into `dst`, eg.
//   for (size_t i = 0; i < triangle_count*3; i++) { dst[i] = my_index_data[triangle_offset*3 + i]; }
typedef void rtk_index_func(void *user, uint32_t *dst, size_t triangle_offset, size_t triangle_count);

typedef struct rtk_index_desc {
	rtk_buffer_desc buffer; // < Default type: RTK_TYPE_U32
	rtk_index_func *func;   // < Custom callback to fetch indices (overrides `buffer`)
	void *func_user;        // < User context pointer passed to `func`
} rtk_index_desc;

typedef struct rtk_mesh_desc {
	size_t num_triangles;       // < Number of triangles in the mesh
	rtk_vertex_desc vertex;     // < Vertex position data, 3*3 floats per triangle
	rtk_index_desc index;       // < Optional vertex index data, 3 uints per triangle
	rtk_user_data user_data;    // < Arbitrary user data attached to the mesh
} rtk_mesh_desc;

typedef struct rtk_primitive_desc {
	rtk_bounds bounds;       // < Axis-aligned bounding box
	rtk_user_data user_data; // < Arbitrary user data attached to the primitive
} rtk_primitive_desc;

typedef struct rtk_scene_desc {

	const rtk_mesh_desc *meshes;
	size_t num_meshes;

	const rtk_primitive_desc *primitives;
	size_t num_primitives;

} rtk_scene_desc;

// Opaque handle to a build
typedef struct rtk_build rtk_build;
typedef struct rtk_build_ctx rtk_build_ctx;
typedef void rtk_task_fn(const struct rtk_task *task, rtk_build_ctx *ctx);

// Task for threading scene building
typedef struct rtk_task {
	rtk_build *build; // < Build operation for the task
	double cost;      // < Estimated cost of this operation in approximate milliseconds
	const char *name; // < Name of the task for debugging/profiling

	// Internal data
	rtk_task_fn *internal_fn;
	size_t internal_index;
} rtk_task;

// -- Simple build API

// Create a scene from the description `desc`. Returns NULL if scene could not be allocatted.
rtk_scene *rtk_build_scene(const rtk_scene_desc *desc);

// Free a scene returned by `rtk_build_scene()` / `rtk_finish_build()` (without custom allocation)
void rtk_free_scene(rtk_scene *scene);

// -- Advanced build API: Allows multi-threaded building and/or providing your own memory allocations

// Example 1: Custom allocation without threads
//
//   size_t build_size = rtk_get_build_size(desc);
//   void *build_data = malloc(build_size);
//
//   rtk_build *build = rtk_start_build(desc, NULL, build_data, build_size);
//
//   size_t scene_size = rtk_get_scene_size(desc);
//   void *scene_data = malloc(scene_size);
//
//   rtk_scene *scene = rtk_finish_build(build, scene_data, scene_size);
//   free(build_data);

// Example 2: Threads without allocation, using an imaginary blocking `work_queue<>`
//
//   template <typename T> struct work_queue {
//       void push(const T *tasks, size_t num); // Push tasks to the queue
//       bool get(T *t); // Fetch the next item or block until finished (return `false`)
//       void finish();  // Broadcast finish, `get()` will return `false` after this
//   }
//
//   work_queue<rtk_task> queue;
//
//   void worker_thread() {
//       rtk_task task;
//       while (queue.get(&task)) {
//           rtk_task followup[64];
//           size_t num_followup = rtk_run_task(&task, followup, 64);
//           if (num_followup == SIZE_MAX) {
//               queue.finish();
//           } else {
//				queue.push(followup, num_followup);
//           }
//       }
//   }
//
//   rtk_task first_task;
//   rtk_build *build = rtk_start_build(desc, &first_task, NULL, 0);
//   queue.push(&first_task, 1);
//
//   fork_threads(&worker_thread, num_threads);
//
//   rtk_scene *scene = rtk_finish_build(build, NULL, 0);

// Query the required temporary memory to build `desc`
size_t rtk_get_build_size(const rtk_scene_desc *desc);

// Start a build operation. If you pass `opt_first_task` you can run the tasks
// in multiple threads using `rtk_run_task()`. If you pass `opt_buffer` the library
// will use that allocation as temporary memory (use `rtk_get_build_size()` for size)
// The pointers in `desc` (but not `&desc` itself) and `opt_buffer` must remain
// valid until `rtk_finish_build()` has returned!
rtk_build *rtk_start_build(const rtk_scene_desc *desc, rtk_task *opt_first_task, const void *opt_buffer, size_t opt_buffer_size);

// Run a task to advance the build.
// TODO: Document when the API is solid!
size_t rtk_run_task(const rtk_task *task, rtk_task *followup_queue, size_t followup_queue_size);

// Query the required final allocation size for the built scene. The build operation
// must have finished before this is called.
size_t rtk_get_scene_size(rtk_build *build);

// Produce a final scene from the build. If you pass `opt_buffer` the scene will
// be written to that buffer, don't call `rtk_free_scene()` in that case!
// If you didn't provide temporary memory for `build` it will be freed automatically
// regardless of whether this function succeeds or not.
rtk_scene *rtk_finish_build(rtk_build *build, const void *opt_buffer, size_t opt_buffer_size);

// -- Ray tracing API

// Trace a ray into the scene to find the closest hit information in `hit`.
// Returns `true` if hit anything.
bool rtk_trace_ray(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit);

// Trace a ray into the scene gathering up to `max_hits` hits, returns the number
// of actual hits.
size_t rtk_trace_ray_all_hits(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hits, size_t max_hits);

// Trace a ray into the scene, returning `true` if the ray hit anything.
bool rtk_trace_ray_any_hit(const rtk_scene *scene, const rtk_ray *ray);

// Modify `hit` to adjust the closest hit information, return the new maximum t value:
//   - `max_t`: Ignore this hit
//   - `hit->t`: Accept this hit
//   - `+RTK_INF`: Visit all hits
//   - `-RTK_INF`: Stop visiting any hits
//   - any value: Custom intersection distance eg. for primitive
typedef rtk_real rtk_visit_func(void *user, const rtk_ray *ray, rtk_hit *hit, rtk_real max_t);

// Trace a ray with callback on every intersection, returns the final maximum t value.
rtk_real rtk_visit_ray(const rtk_scene *scene, const rtk_ray *ray, rtk_visit_func *func, void *user);

#ifdef __cplusplus
}
#endif

#endif
