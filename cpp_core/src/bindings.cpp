#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Automatic conversion between std::vector/std::optional and Python list/None
#include <pybind11/eigen.h>     // Automatic zero-copy conversion between Eigen::Vector2f and NumPy array

#include <deco_core/geometry.hpp>
#include <deco_core/collision.hpp>

namespace py = pybind11;

// PYBIND11_MODULE(module_name, module_handle)
// The first argument must match the CMake target name exactly (deco_core_cpp).
PYBIND11_MODULE(deco_core_cpp, m) {
    m.doc() = "DECO-SWARM: High-Performance C++ Simulation Core";

    // -------------------------------------------------------------
    // 1. Data Structures Binding
    // -------------------------------------------------------------

    // Segment2D struct: binds p1 and p2 endpoints
    py::class_<deco::Segment2D>(m, "Segment2D")
        .def(py::init<deco::Vec2, deco::Vec2>(), py::arg("p1"), py::arg("p2"))
        .def_readwrite("p1", &deco::Segment2D::p1)
        .def_readwrite("p2", &deco::Segment2D::p2)
        .def("__repr__", [](const deco::Segment2D& s) {
            return "<Segment2D [(" + std::to_string(s.p1.x()) + ", " + std::to_string(s.p1.y()) +
                   ") -> (" + std::to_string(s.p2.x()) + ", " + std::to_string(s.p2.y()) + ")]>";
        });

    // RayHit struct: returns endpoint coordinates, normalized distance, and wall hit flag
    py::class_<deco::RayHit>(m, "RayHit")
        .def_readonly("endpoint", &deco::RayHit::endpoint)
        .def_readonly("norm_dist", &deco::RayHit::norm_dist)
        .def_readonly("has_hit", &deco::RayHit::has_hit)
        .def("__repr__", [](const deco::RayHit& r) {
            return "<RayHit norm_dist=" + std::to_string(r.norm_dist) +
                   " has_hit=" + (r.has_hit ? "True" : "False") + ">";
        });

    // EgoPolarTarget struct: normalized distance and relative angle heading
    py::class_<deco::EgoPolarTarget>(m, "EgoPolarTarget")
        .def_readonly("norm_dist", &deco::EgoPolarTarget::norm_dist)
        .def_readonly("norm_angle", &deco::EgoPolarTarget::norm_angle)
        .def("__repr__", [](const deco::EgoPolarTarget& t) {
            return "<EgoPolarTarget dist=" + std::to_string(t.norm_dist) +
                   " angle=" + std::to_string(t.norm_angle) + ">";
        });

    // CollisionResult struct: circle position after collision sliding and boolean flag
    py::class_<deco::CollisionResult>(m, "CollisionResult")
        .def_readonly("new_position", &deco::CollisionResult::new_position)
        .def_readonly("collided", &deco::CollisionResult::collided)
        .def("__repr__", [](const deco::CollisionResult& c) {
            return "<CollisionResult collided=" + std::string(c.collided ? "True" : "False") + ">";
        });

    // -------------------------------------------------------------
    // 2. Geometry Mathematical Functions Binding
    // -------------------------------------------------------------

    m.def("wrap_angle_deg", &deco::wrap_angle_deg,
          "Clamps angle in degrees to [0.0, 360.0)",
          py::arg("angle_deg"));

    m.def("shortest_angle_diff_deg", &deco::shortest_angle_diff_deg,
          "Computes shortest angular difference in [-180.0, +180.0] degrees",
          py::arg("target_deg"), py::arg("source_deg"));

    m.def("distance", &deco::distance,
          "Euclidean distance between two 2D points",
          py::arg("p1"), py::arg("p2"));

    m.def("intersect_ray_segment", &deco::intersect_ray_segment,
          "Fast parametric ray-segment intersection. Returns RayHit or None",
          py::arg("ray_origin"), py::arg("ray_dir"), py::arg("max_dist"), py::arg("segment"));

    m.def("compute_ego_polar_target", &deco::compute_ego_polar_target,
          "Computes normalized relative polar target (norm_dist, norm_angle)",
          py::arg("agent_pos"), py::arg("agent_angle_deg"), py::arg("target_pos"), py::arg("max_map_distance"));

    // -------------------------------------------------------------
    // 3. Collision Resolution & Batch Raycast Binding
    // -------------------------------------------------------------

    m.def("resolve_circle_segment", &deco::resolve_circle_segment,
          "Resolves circle collision against a single line segment",
          py::arg("circle_center"), py::arg("radius"), py::arg("segment"));

    m.def("resolve_circle_segments", &deco::resolve_circle_segments,
          "Resolves circle collision against a list of line segments with multi-pass corner relaxation",
          py::arg("circle_center"), py::arg("radius"), py::arg("segments"));

    m.def("cast_rays", &deco::cast_rays,
          "Batched multi-raycast sweep across FOV",
          py::arg("origin"), py::arg("heading_deg"), py::arg("fov_deg"),
          py::arg("num_rays"), py::arg("max_dist"), py::arg("segments"));
}
