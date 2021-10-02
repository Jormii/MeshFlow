class Vec2:

    def __init__(self, x, y, vec_type=float):
        self.x = vec_type(x)
        self.y = vec_type(y)

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


class Vec3:

    def __init__(self, x, y, z, vec_type=float):
        self.x = vec_type(x)
        self.y = vec_type(y)
        self.z = vec_type(z)

    def __repr__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


class Vec4:

    def __init__(self, x, y, z, w, vec_type=float):
        self.x = vec_type(x)
        self.y = vec_type(y)
        self.z = vec_type(z)
        self.w = vec_type(w)

    def __repr__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.z, self.w)
