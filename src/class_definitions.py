class CharDef:
    def __init__(self, uid, lid, compsLength, compIds, preciseDef=False, boxes=None):
        self.uid = uid
        self.lid = lid
        self.compsLen = compsLength
        self.compIds = compIds
        self.preciseDef = preciseDef
        self.boxes = boxes

    def __str__(self):
        return str((self.uid, self.lid, self.compIds, self.preciseDef))

    def __repr__(self):
        return str(self)

    def setPreciseDef(self, boxes):
        self.preciseDef = True
        self.boxes = boxes

class BoundingBox():
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return str((self.x, self.y, self.dx, self.dy))

    def __repr__(self):
        return str(self)

    def getSize(self):
        return (self.dx,self.dy)

    def getBoxOutline(self):
        return (self.x, self.y, self.x + self.dx, self.y + self.dy)

class ArrayCollection:
    def __init__(self, uid, style, character=None, roughDefinition=None, lossMap=None):
        self.uid = uid
        self.style = style
        self.character = character
        self.roughDefinition = roughDefinition
        self.lossMap = lossMap

    def addCharacter(self, character):
        self.character = character

    def addRoughDef(self, roughDefinition):
        self.roughDefinition = roughDefinition

    def addLossMap(self, lossMap):
        self.lossMap = lossMap

    def isComplete(self):
        return self.character is not None and self.roughDefinition is not None

    def printIncomplete(self):
        if self.character is None:
            print("Character " + self.uid + " is missing its character array")
        if self.roughDefinition is None:
            print("Character " + self.uid + " is missing its roughDefinition array")
        if self.lossMap is None:
            print("Character " + self.uid + " is missing its lossMap array")

    def asList(self):
        return [self.uid, self.style, self.character, self.roughDefinition, self.lossMap]