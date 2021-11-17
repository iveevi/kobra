#include "include/mesh/cuboid.hpp"

namespace mercury {

namespace mesh {

glm::vec3 _project(const glm::vec3 &v, const glm::vec3 &u)
{
        return u * glm::dot(v, u) / glm::dot(u, u);
}

// Wireframe cuboid
SVA3 wireframe_cuboid(const glm::vec3 &center, const glm::vec3 &size,
                const glm::vec3 &up)
{
        glm::vec3 right = glm::cross(up, glm::vec3(0.0f, 0.0f, 1.0f));
        return wireframe_cuboid(center, size, up, right);
}

SVA3 wireframe_cuboid(const glm::vec3 &center, const glm::vec3 &size,
                const glm::vec3 &up, const glm::vec3 &right)
{
        glm::vec3 nup = glm::normalize(up);
        glm::vec3 nright = glm::normalize(right);
        glm::vec3 nforward = glm::normalize(glm::cross(nup, nright));

        glm::vec3 xdir = nright * size.x/2.0f;
        glm::vec3 ydir = nup * size.y/2.0f;
        glm::vec3 zdir = nforward * size.z/2.0f;

        // TODO: some way to convert to a list of faces
        return SVA3({
                // Front face
                center + xdir + ydir + zdir,
                center - xdir + ydir + zdir,
                center - xdir - ydir + zdir,
                center + xdir - ydir + zdir,
                center + xdir + ydir + zdir,
                
                // Left face
                center + xdir + ydir - zdir,
                center + xdir - ydir - zdir,
                center + xdir - ydir + zdir,
                center + xdir + ydir + zdir,
                center + xdir + ydir - zdir,

                // Top face
                center - xdir + ydir - zdir,
                center - xdir + ydir + zdir,
                center + xdir + ydir + zdir,
                center + xdir + ydir - zdir,
                center - xdir + ydir - zdir,

                // Back face
                center - xdir - ydir - zdir,
                center + xdir - ydir - zdir,
                center + xdir + ydir - zdir,
                center - xdir + ydir - zdir,
                center - xdir - ydir - zdir,

                // Right face
                center - xdir - ydir + zdir,
                center - xdir + ydir + zdir,
                center - xdir + ydir - zdir,
                center - xdir - ydir - zdir,
                center - xdir - ydir + zdir,

                // Bottom face
                center + xdir - ydir + zdir,
                center + xdir - ydir - zdir,
                center - xdir - ydir - zdir,
                center - xdir - ydir + zdir,
                center + xdir - ydir + zdir
        });
}

}

}