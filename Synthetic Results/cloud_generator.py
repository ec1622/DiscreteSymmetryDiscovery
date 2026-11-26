def cube(n):
    return torch.rand(n, 3) * 2 - 1

def cuboid(n):
    return torch.cat((torch.rand(n, 2) * 2 - 1, torch.rand(n, 1) * 4 - 2), dim=1)

def triangular_prism(n):
    points = []
    for _ in range(n):
        x = torch.rand(1) * np.sqrt(3) - (np.sqrt(3) / 2)
        y = torch.rand(1) * (3 / 2) - 0.5
        while y > min(-np.sqrt(3) * x + 1, np.sqrt(3) * x + 1):
            x = torch.rand(1) * np.sqrt(3) - (np.sqrt(3) / 2)
            y = torch.rand(1) * (3 / 2) - 0.5
        z = torch.rand(1) * 4 - 2
        points.append([x, y, z])
    return torch.tensor(points)

def pentagonal_prism(n):
    a = (1 + np.sqrt(5)) / 4
    sin36 = np.sqrt(10 - 2 * np.sqrt(5)) / 4
    cos18 = np.sqrt(10 + 2 * np.sqrt(5)) / 4
    sin18 = (np.sqrt(5) - 1) / 4
    points = []
    for _ in range(n):
        while True:
            x, y = 2 * torch.rand(2) - 1.0
            if (-sin36 * x + a * y <= a and
                -cos18 * x - sin18 * y <= a and
                y >= -a and
                cos18 * x - sin18 * y <= a and
                sin36 * x + a * y <= a):
                break
        z = 4 * torch.rand(1) - 2.0
        points.append([x, y, z])
    return torch.tensor(points)

def hexagonal_prism(n):
    points = []
    for _ in range(n):
        while True:
            x, y = 2 * torch.rand(2) - 1.0
            if torch.abs(y) <= np.sqrt(3) / 2 and np.sqrt(3) * max(x - 1, -x - 1) <= y <= np.sqrt(3) * min(x + 1, -x + 1):
                break
        z = 4 * torch.rand(1) - 2.0
        points.append([x, y, z])
    return torch.tensor(points)

def octagonal_prism(n):
    points = []
    for _ in range(n):
        while True:
            x, y = 2 * torch.rand(2) - 1.0
            if torch.abs(x) + torch.abs(y) <= (2.0 + math.sqrt(2.0)) / (1.0 + math.sqrt(2.0)):
                break
        z = 4 * torch.rand(1) - 2.0
        points.append([x, y, z])
    return torch.tensor(points)

def tetrahedron(n):
    points = []
    for _ in range(n):
        while True:
            x, y, z = 2 * torch.rand(3) - 1.0
            if x + y - z <= 1 and x - y + z <= 1 and -x + y + z <= 1 and x + y + z >= -1:
                break
        points.append([x, y, z])
    return torch.tensor(points)

def octahedron(n):
    points = []
    for _ in range(n):
        while True:
            x, y, z = 2 * torch.rand(3) - 1.0
            if torch.abs(x) + torch.abs(y) + torch.abs(z) <= 1.0:
                break
        points.append([x, y, z])
    return torch.tensor(points)

def dodecahedron(n):
    phi = (1 + np.sqrt(5)) / 2
    points = []
    for _ in range(n):
        while True:
            x, y, z = 2 * phi * torch.rand(3) - phi
            if (phi * torch.abs(y) + torch.abs(z) <= phi**2 and
                torch.abs(x) + phi * torch.abs(z) <= phi**2 and
                phi * torch.abs(x) + torch.abs(y) <= phi**2):
                break
        points.append([x, y, z])
    return torch.tensor(points)

def icosahedron(n):
    phi = (1 + np.sqrt(5)) / 2
    points = []
    for _ in range(n):
        while True:
            x, y, z = 2 * phi * torch.rand(3) - phi
            if (torch.abs(x) + torch.abs(y) + torch.abs(z) <= phi**2 and
                (phi - 1) * torch.abs(x) + phi * torch.abs(z) <= phi**2 and
                phi * torch.abs(x) + (phi - 1) * torch.abs(y) <= phi**2 and
                phi * torch.abs(y) + (phi - 1) * torch.abs(z) <= phi**2):
                break
        points.append([x, y, z])
    return torch.tensor(points)

def circular_prism(n):
    points = []
    for _ in range(n):
        x, y = 2 * torch.rand(2) - 1.0
        while x**2 + y**2 > 1:
            x, y = 2 * torch.rand(2) - 1.0
        z = 4 * torch.rand(1) - 2.0
        points.append([x, y, z])
    return torch.tensor(points)

def square(n):
    return torch.rand(n, 2) * 2 - 1

def octagon(n):
    points = []
    for _ in range(n):
        while True:
            x, y = 2 * torch.rand(2) - 1.0
            if torch.abs(x) + torch.abs(y) <= (2.0 + math.sqrt(2.0)) / (1.0 + math.sqrt(2.0)):
                break
        points.append([x, y])
    return torch.tensor(points)

def star(n):
    points = []
    for _ in range(n):
        while True:
            x, y = 2 * math.sqrt(10.0) * torch.rand(2) - math.sqrt(10.0)
            if math.sqrt(x**2 + y**2) <= torch.abs(torch.pi / 3 + (2 / 3) * torch.arccos(torch.cos(5 * torch.atan2(y, x)))):
                break
        points.append([x, y])
    return torch.tensor(points)
