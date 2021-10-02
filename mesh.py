import numpy as np


class Mesh:

    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.triangles = []
        self.quads = []
        self.colors = []
        self.normals = []
        self.uvs = []

    def as_numpy(self):
        copy = Mesh(self.name)

        copy.vertices = np.array(self.vertices)
        copy.triangles = np.array(self.triangles)
        copy.quads = np.array(self.quads)
        copy.colors = np.array(self.colors)
        copy.normals = np.array(self.normals)
        copy.uvs = np.array(self.uvs)

        return copy
