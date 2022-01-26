#ifndef CAMERA_H_
#define CAMERA_H_

// Engine headers
#include "transform.hpp"

// Ray structure
struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
};

// Camera properties
struct Tunings {
        float fov;
        float aspect;
        float near;
        float far;

        Tunings() : fov(0.0f), aspect(0.0f),
                        near(0.0f), far(0.0f) {}

        Tunings(float fov, float width, float height)
                        : fov(fov), aspect(width / height),
                        near(0.1f), far(100.0f) {}
};

// Camera class
class Camera {
        // Transforms
        // TODO: inhert
        Transform transform;
        Transform camera_to_raster;
        Transform raster_to_camera;

        // Camera parameters
        Tunings tunings;
        
        // Screen space differentials
        glm::vec3 dx;
        glm::vec3 dy;

        // Generate perspective transform
        static Transform _perspective(float fov, float aspect, float near, float far) {
                glm::mat4 perspective = glm::perspective(fov, aspect, near, far);
                return Transform(perspective);
        }
public:
        // Constructor with position, front and up vectors
        Camera(glm::vec3 position, glm::vec3 front, glm::vec3 up, const Tunings& tns)
                        : tunings(tns) {
                // Set camera position
                transform.translate(position);

                // Set camera orientation
                transform.look_at(front, up);

                // Set camera to raster transform
                camera_to_raster = _perspective(
                        tunings.fov, tunings.aspect,
                        tunings.near, tunings.far
                );

                // Set raster to camera Transform
                raster_to_camera = camera_to_raster.inverse();

                // Calculate screen space differentials
                glm::vec3 ux {1.0f, 0.0f, 0.0f};
                glm::vec3 uy {0.0f, 1.0f, 0.0f};

                dx = raster_to_camera.mul_pt(ux);
                dy = raster_to_camera.mul_pt(uy);
        }

        // Generate a ray from the camera to the given pixel
        Ray ray(float nx, float ny) const {
                glm::vec3 film_pt {nx, ny, 0.0f};
                glm::vec3 camera_pt = raster_to_camera(film_pt);
                return Ray {
                        glm::vec3 {0.0f},
                        glm::normalize(camera_pt)
                };
        }
};

#endif
