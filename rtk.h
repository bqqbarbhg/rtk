#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RTK_INF (3.402823e+38f)

typedef float rtk_real;

typedef struct rtk_vec3 {
	union {
		struct {
			rtk_real x, y, z;
		};
		rtk_real v[3];
	};
} rtk_vec3;

typedef struct rtk_vertex {
	rtk_vec3 position;
	uint32_t index;
} rtk_vertex;

typedef struct rtk_ray {
	rtk_vec3 origin;
	rtk_vec3 direction;
	rtk_real min_t;
	rtk_real max_t;
} rtk_ray;

typedef struct rtk_hit {
	rtk_real t;
	rtk_real u;
	rtk_real v;
	rtk_vertex vertex[3];
	uint32_t mesh_index;
	uint32_t triangle_index;
} rtk_hit;

typedef enum rtk_type {
	RTK_TYPE_DEFAULT,
	RTK_TYPE_F32,
	RTK_TYPE_F64,
	RTK_TYPE_REAL,
	RTK_TYPE_U16,
	RTK_TYPE_U32,
} rtk_type;

typedef struct rtk_buffer {
	const void *data;
	size_t stride;
	rtk_type type;
} rtk_buffer;

typedef struct rtk_mesh rtk_mesh;
typedef void rtk_position_callback_fn(void *user, const rtk_mesh *mesh, rtk_vec3 *dst, const uint32_t *indices, size_t count);
typedef void rtk_index_callback_fn(void *user, const rtk_mesh *mesh, uint32_t *dst, size_t offset, size_t count);

struct rtk_mesh {
	void *user;
	size_t num_triangles;

	rtk_buffer position; // Default type: REAL
	rtk_buffer index; // Default type: U32

	rtk_position_callback_fn *position_cb;
	void *position_cb_user;

	rtk_index_callback_fn *index_cb;
	void *index_cb_user;
};

typedef struct rtk_scene {
	char magic[8];
	uint16_t endian;
	uint8_t sizeof_real;
	uint8_t pad_0;
	uint32_t version;
	uint32_t pad_1;
	uint64_t size_in_bytes;
	uint64_t node_offset;
	uint64_t leaf_offset;
	uint64_t vertex_offset;
} rtk_scene;

typedef struct rtk_build rtk_build;
typedef struct rtk_task rtk_task;
typedef struct rtk_task_ctx rtk_task_ctx;

typedef void rtk_log_fn(void *user, rtk_build *build, const char *str);

typedef struct rtk_scene_desc {

	const rtk_mesh *meshes;
	size_t num_meshes;

	rtk_log_fn *log_fn;
	void *log_user;

} rtk_scene_desc;


typedef void rtk_task_fn(const rtk_task *task, rtk_task_ctx *ctx);
struct rtk_task {
	rtk_build *build;
	rtk_task_fn *fn;
	double cost;
	size_t index;
	uintptr_t arg;
};

typedef bool rtk_filter_fn(void *user, const rtk_ray *ray, const rtk_hit *hit);

rtk_build *rtk_start_build(const rtk_scene_desc *desc, rtk_task *first_task);
size_t rtk_run_task(const rtk_task *task, rtk_task *queue, size_t queue_size);

size_t rtk_get_build_size(const rtk_build *build);
rtk_scene *rtk_finish_build_to(rtk_build *build, void *buffer, size_t size);
rtk_scene *rtk_finish_build(rtk_build *build);

rtk_scene *rtk_build_scene(const rtk_scene_desc *desc);
void rtk_free_scene(rtk_scene *scene);

bool rtk_trace_ray(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit);
bool rtk_trace_ray_filter(const rtk_scene *scene, const rtk_ray *ray, rtk_hit *hit, rtk_filter_fn *filter, void *filter_user);

#ifdef __cplusplus
}
#endif

