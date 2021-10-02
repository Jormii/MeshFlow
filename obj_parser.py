from mesh import Mesh

# .obj fields
OBJ_COMMENT_TOKEN = "#"
OBJ_MESH_NAME_TOKEN = "o"
OBJ_VERTEX_TOKEN = "v"
OBJ_UV_TOKEN = "vt"
OBJ_NORMAL_TOKEN = "vn"
OBJ_SMOOTH_SHADING_TOKEN = "s"
OBJ_FACE_DATA_TOKEN = "f"

OBJ_FACE_DATA_VALID_FIELDS = [
    OBJ_UV_TOKEN,
    OBJ_NORMAL_TOKEN
]


class ObjParseResult:

    def __init__(self):
        self.obj_meshes = []
        self.comments = []

    def meshes(self):
        m = []
        for obj_mesh in self.obj_meshes:
            m.append(obj_mesh.mesh())

        return m

    def _curr_mesh(self):
        return self.obj_meshes[-1]

    def _new_mesh(self, name):
        self.obj_meshes.append(ObjMesh(name))

    def _comment(self, comment):
        self.comments.append(comment)


class ObjMesh:

    def __init__(self, name):
        self.name = name
        self.fields = []
        self.vertices = []
        self.uvs = []
        self.normals = []
        self.face_data = []

    def mesh(self):
        # Set up variables
        mesh = Mesh(self.name)
        mesh.vertices = self.vertices
        mesh.uvs = [None] * len(self.vertices)
        mesh.normals = [None] * len(self.vertices)

        field_dict = {}
        for i, field_token in enumerate(self.fields):
            if field_token == OBJ_UV_TOKEN:
                field_dict[i] = (self.uvs, mesh.uvs)
            elif field_token == OBJ_NORMAL_TOKEN:
                field_dict[i] = (self.normals, mesh.normals)

        # Write data
        for face_data in self.face_data:
            quad = []

            for vertex_data in face_data:
                vertex_index = vertex_data[0]
                other_data = vertex_data[1:]

                quad.append(vertex_index)
                for i, data in enumerate(other_data):
                    r, w = field_dict[i]
                    if w[vertex_index] is not None:
                        continue    # TODO: Possible?

                    w[vertex_index] = r[data]

            mesh.quads.append(quad)

        return mesh.as_numpy()

    def _field(self, field_token):
        if field_token not in self.fields:
            self.fields.append(field_token)

    def _vertex(self, vertex):
        self.vertices.append(vertex)

    def _uv(self, uv):
        self.uvs.append(uv)

    def _normal(self, normal):
        self.normals.append(normal)

    def _face_data(self, face_data):
        self.face_data.append(face_data)


def parse_obj(obj_path):
    content = ObjParseResult()
    obj_file = open(obj_path, "r")
    line = obj_file.readline()
    line_count = 1

    try:
        while line:
            _parse_line(content, line)
            line = obj_file.readline()
            line_count += 1
    except Exception as e:
        print("Unknown field. Exception caused by line {}".format(line_count))
        print(e)
    finally:
        obj_file.close()

    return content


def _parse_line(obj_parse, line):
    if len(line) == 0:
        # Ignore empty lines
        return

    line_split = line.strip().split(" ")
    field_token = line_split[0]
    if field_token in OBJ_FACE_DATA_VALID_FIELDS:
        obj_parse._curr_mesh()._field(field_token)
    PARSE_FUNCS[field_token](obj_parse, line_split[1:])


def _parse_comment(obj_parse, line_split):
    obj_parse._comment(" ".join(line_split))


def _parse_mesh_name(obj_parse, line_split):
    mesh_name = line_split[0]
    obj_parse._new_mesh(mesh_name)


def _parse_vertex(obj_parse, line_split):
    vertex = []
    for v in line_split:
        vertex.append(float(v))

    obj_parse._curr_mesh()._vertex(vertex)


def _parse_uv(obj_parse, line_split):
    uv = []
    for v in line_split:
        uv.append(float(v))

    obj_parse._curr_mesh()._uv(uv)


def _parse_normal(obj_parse, line_split):
    normal = []
    for v in line_split:
        normal.append(float(v))

    obj_parse._curr_mesh()._normal(normal)


def _parse_smooth(obj_parse, line_split):
    # TODO
    return


def _parse_face_data(obj_parse, line_split):
    assert len(line_split) == 4, "Only supports quads"

    face_data = []
    for f in line_split:
        face_data.append([])
        sub_split = f.split("/")
        for v in sub_split:
            face_data[-1].append(int(v) - 1)

    obj_parse._curr_mesh()._face_data(face_data)


PARSE_FUNCS = {
    OBJ_COMMENT_TOKEN: _parse_comment,
    OBJ_MESH_NAME_TOKEN: _parse_mesh_name,
    OBJ_VERTEX_TOKEN: _parse_vertex,
    OBJ_UV_TOKEN: _parse_uv,
    OBJ_NORMAL_TOKEN: _parse_normal,
    OBJ_SMOOTH_SHADING_TOKEN: _parse_smooth,
    OBJ_FACE_DATA_TOKEN: _parse_face_data
}
