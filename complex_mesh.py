import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as fill_polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Edge:

    def __init__(self, vi, vj, quad):
        self.vi = vi
        self.vj = vj
        self.quad = quad

    def flipped(self):
        return Edge(self.vj, self.vi, self.quad)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        return self.vi == other.vi and self.vj == other.vj

    def __hash__(self):
        return hash((self.vi, self.vj))

    def __repr__(self):
        return "E ({}, {}) -> {}".format(self.vi, self.vj, self.quad)


class Flow:

    def __init__(self, ei, ej):
        self.ei = ei
        self.ej = ej

    def __eq__(self, other):
        if not isinstance(other, Flow):
            return False

        return self.ei == other.ei and self.ej == other.ej

    def __hash__(self):
        return hash((self.ei, self.ej))

    def __repr__(self):
        return "({}, {})".format(self.ei, self.ej)


class ComplexMesh:

    def __init__(self, mesh):
        # TODO: Allow tris
        assert len(mesh.triangles) == 0 and len(
            mesh.quads) != 0, "Mesh constains triangles or is empty"
        assert len(mesh.uvs) != 0, "UVs are mandatory"

        self.mesh = mesh
        self._initialize()
        self._initialize_boundary_edges()
        self._initialize_oposed_edges()
        self._initialized_ring_edges()

    def _initialize(self):
        # Build edges and membership
        self.edges = []
        self.quad_edge_membership = {}
        self.vertex_edge_membership = {}

        for quad_index, quad in enumerate(self.mesh.quads):
            quad_edges = [
                Edge(quad[0], quad[1], quad_index),
                Edge(quad[1], quad[2], quad_index),
                Edge(quad[2], quad[3], quad_index),
                Edge(quad[3], quad[0], quad_index)
            ]

            base_size = len(self.edges)
            self.edges.extend(quad_edges)
            self.quad_edge_membership[quad_index] = []

            for i, edge in enumerate(quad_edges):
                edge_index = base_size + i
                self.quad_edge_membership[quad_index].append(edge_index)

                for vertex_index in [edge.vi, edge.vj]:
                    if vertex_index not in self.vertex_edge_membership:
                        self.vertex_edge_membership[vertex_index] = []

                    self.vertex_edge_membership[vertex_index].append(
                        edge_index)

    def _initialize_boundary_edges(self):
        # Build boundary and non-boundary edges
        boundary_edges = {}
        self.non_boundary_edges = {}
        for i, edge in enumerate(self.edges):
            flipped = edge.flipped()
            if flipped in boundary_edges:
                other_edge_index = boundary_edges[flipped]
                del boundary_edges[flipped]

                self.non_boundary_edges[i] = other_edge_index
                self.non_boundary_edges[other_edge_index] = i
            else:
                boundary_edges[edge] = i

        self.boundary_edges = set(boundary_edges.values())

    def _initialize_oposed_edges(self):
        self.oposed_edges = {}

        for edge_list in self.quad_edge_membership.values():
            for edge_index in edge_list:
                if edge_index not in self.oposed_edges:
                    self.oposed_edges[edge_index] = []

                oposed_edge_index = self._find_oposed_edge(
                    edge_index, edge_list)
                self.oposed_edges[edge_index].append(oposed_edge_index)

                if oposed_edge_index in self.non_boundary_edges:
                    self.oposed_edges[edge_index].append(
                        self.non_boundary_edges[oposed_edge_index])

    def _find_oposed_edge(self, edge_index, edge_list):
        edge = self.edges[edge_index]

        for other_edge_index in edge_list:
            if edge_index == other_edge_index:
                continue

            other_edge = self.edges[other_edge_index]
            indexes_set = set([edge.vi, edge.vj, other_edge.vi, other_edge.vj])
            if len(indexes_set) == 4:
                # Means no vertices are shared. Therefore they are oposed
                return other_edge_index

        exit("No oposed edge found. This is impossible")

    def _initialized_ring_edges(self):
        self.ring_edges = {}

        for edge_index, edge in enumerate(self.edges):
            # Get neighbor edges
            neighbor_edges = set()
            for vertex_index in [edge.vi, edge.vj]:
                neighbor_edges.update(
                    self.vertex_edge_membership[vertex_index])

            # Get quads involved with the original edge
            involved_quads = {edge.quad}
            if edge_index in self.non_boundary_edges:
                involved_quads.add(self.non_boundary_edges[edge_index])

            # Ring edges are those rings that do not belong to the involved quads
            for other_edge_index in neighbor_edges:
                other_edge = self.edges[other_edge_index]
                if other_edge.quad not in involved_quads:
                    if edge_index not in self.ring_edges:
                        self.ring_edges[edge_index] = []

                    self.ring_edges[edge_index].append(other_edge_index)

    def calculate_flow(self):
        f_boundary, f_boundary_seed = self._boundary_edges_flow()

        textures = []
        for i, (f, seed) in enumerate(zip(f_boundary, f_boundary_seed)):
            print("{} > [{}, {}]".format(i, len(f), self.edges[seed]))
            textures.append(self._write_flow(f, seed))

        return textures

    def _boundary_edges_flow(self):
        f_boundary = []
        f_boundary_seed = []
        for edge_index in self.boundary_edges:
            if not self._skip_calculation(edge_index, f_boundary):
                f_boundary.append(self._calculate_edge_flow(edge_index))
                f_boundary_seed.append(edge_index)

        return f_boundary, f_boundary_seed

    def _skip_calculation(self, edge_index, calculated_f):
        for f in calculated_f:
            for flow in f:
                if edge_index in [flow.ei, flow.ej]:
                    return True

        return False

    def _calculate_edge_flow(self, edge_index):
        flow = set()
        visited_edges = set()
        edges_to_check = [edge_index]

        while len(edges_to_check) != 0:
            index = edges_to_check.pop()

            # Check and update visited_edges
            if index in visited_edges:
                continue

            visited_edges.add(index)

            # Calculate flow and expand edges to check
            # Same polygon
            oposed = self.oposed_edges[index]
            # TODO: Ring edges likely wrong
            ring = self.ring_edges.get(None, list())
            for edge_list in [oposed, ring]:
                for other_edge_index in edge_list:
                    flow.add(Flow(
                        min(index, other_edge_index),
                        max(index, other_edge_index)
                    ))
                    edges_to_check.append(other_edge_index)

        return flow

    def _write_flow(self, f, f_seed):
        size = 1080
        shape = (size, size, 3)
        texture = np.zeros(shape, dtype="float32")

        for flow in f:
            # Retrive and calculate information to write
            begin_midpoint = self.midpoint(flow.ei)
            end_midpoint = self.midpoint(flow.ej)

            v = end_midpoint - begin_midpoint
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                # TODO: Shouldn't be necessary
                continue

            v_u = v / np.linalg.norm(v)

            # Retrieve uvs to fill
            quad_index = self._retrieve_quad(flow.ei, flow.ej)
            quad = self.mesh.quads[quad_index]
            uvs = self.mesh.uvs[quad]

            # Write data
            print(
                "- [{}, {}] = {}".format(self.edges[flow.ei], self.edges[flow.ej], v_u))

            uvs_texture = np.floor(size * uvs).astype(int)
            polygon = Polygon(uvs_texture)
            polygon_vertices = polygon.exterior.xy

            X, Y = fill_polygon(
                polygon_vertices[1], polygon_vertices[0], (size, size))
            texture[X, Y, :3] = v_u

            if False:
                plt.plot(uvs_texture[:, 0], uvs_texture[:, 1], "+")
                plt.imshow(texture, origin="lower")
                plt.show()
                plt.close()

        return self._fix_negative(np.flipud(texture))

    def _retrieve_quad(self, ei, ej):
        edge_i = self.edges[ei]
        edge_j = self.edges[ej]

        # If they are part of the same quad, return
        if edge_i.quad == edge_j.quad:
            return edge_i.quad

        # Obtain the common quad
        involved_faces = {
            edge_i.quad: 1,
            edge_j.quad: 1
        }
        for edge_index in [ei, ej]:
            if edge_index in self.non_boundary_edges:
                shared_edge_index = self.non_boundary_edges[edge_index]
                shared_edge = self.edges[shared_edge_index]

                if shared_edge.quad not in involved_faces:
                    involved_faces[shared_edge.quad] = 0
                involved_faces[shared_edge.quad] += 1

        for quad, count in involved_faces.items():
            if count == 2:
                return quad

        exit("Couldn't find shared quad for edges {} and {}".format(edge_i, edge_j))

    def _fix_negative(self, texture):
        negative_texture = np.zeros(texture.shape, dtype=bool)
        for channel_index in range(texture.shape[-1]):
            channel = texture[:, :, channel_index]
            negative_pixels = np.argwhere(channel < 0)

            I = negative_pixels[:, 0]
            J = negative_pixels[:, 1]
            texture[I, J, channel_index] = np.abs(texture[I, J, channel_index])
            negative_texture[I, J, channel_index] = True

        texture = (255 * texture).astype("uint8")
        negative_texture = 255 * negative_texture.astype("uint8")
        assert len(np.argwhere(texture < 0)
                   ) == 0, "Negative values still present in the texture"

        return texture, negative_texture

    def midpoint(self, edge_index):
        edge = self.edges[edge_index]
        p = self.mesh.vertices[edge.vi]
        q = self.mesh.vertices[edge.vj]

        midpoint = p + 0.5*(q - p)
        return midpoint

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # Plot faces
        for quad in self.mesh.quads:
            ax.add_collection3d(Poly3DCollection(
                [self.mesh.vertices[quad]], alpha=0.5))

            quad_center = np.mean(self.mesh.vertices[quad], axis=0)
            ax.text(quad_center[0], quad_center[1],
                    quad_center[2], repr(quad.tolist()))

        # Plot boundary edges
        for edge_index in self.boundary_edges:
            edge = self.edges[edge_index]
            p = self.mesh.vertices[edge.vi]
            q = self.mesh.vertices[edge.vj]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], "g-")

        # Plot non-boundary edges
        ignore = set()
        for edge_index, other_edge_index in self.non_boundary_edges.items():
            if edge_index in ignore:
                continue

            ignore.add(other_edge_index)
            edge = self.edges[edge_index]
            p = self.mesh.vertices[edge.vi]
            q = self.mesh.vertices[edge.vj]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], "r-")

        # Plot vertices
        for vertex_index, vertex in enumerate(self.mesh.vertices):
            ax.plot(vertex[0], vertex[1], vertex[2], "r.")
            ax.text(vertex[0], vertex[1], vertex[2], repr(vertex_index))

        plt.show()
        plt.close()
