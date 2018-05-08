import vtk
import numpy as np

class VtkPointCloud:

    def __init__(self, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(5)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


class vtkCylinder:

    def __init__(self):
        self.cylinder = vtk.vtkCylinderSource()
        self.cylinder.SetRadius(2.5)
        self.cylinder.SetHeight(50)
        self.cylinder.SetCenter(0, 0, 0)
        self.cylinder.SetResolution(100)
        self.vtkActor = None

    def setCylinder(self):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.cylinder.GetOutputPort())
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def setTransform(self, transformMatrix):
        cylinderTransform = vtk.vtkTransform()
        cylinderTransform.SetMatrix(self.mat2vtk(transformMatrix))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.cylinder.GetOutputPort())
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetUserTransform(cylinderTransform)
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetColor(0.1, 0.2, 0.9)

    def mat2vtk(self, mat):
        obj = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                obj.SetElement(i, j, mat[i, j])
        return obj

class AddToolsTimerCallBack:

    def __init__(self, renderer, transfromList, videoPtCloud, writer):
        self.transformList = transfromList
        self.videoPtCloud = videoPtCloud
        self.renderer = renderer
        self.tool1 = vtk.vtkCylinderSource()
        self.tool1.SetRadius(2.5)
        self.tool1.SetHeight(25)
        self.tool1.SetCenter(0, 0, 0)
        self.tool1.SetResolution(100)
        self.tool1Actor = vtk.vtkActor()
        self.mapperTool1 = vtk.vtkPolyDataMapper()
        self.mapperTool1.SetInputConnection(self.tool1.GetOutputPort())
        self.tool1Actor.GetProperty().SetColor(0.1, 0.2, 0.3)

        self.tool2 = vtk.vtkCylinderSource()
        self.tool2.SetRadius(2.5)
        self.tool2.SetHeight(25)
        self.tool2.SetCenter(0, 0, 0)
        self.tool2.SetResolution(100)
        self.tool2Actor = vtk.vtkActor()
        self.mapperTool2 = vtk.vtkPolyDataMapper()
        self.mapperTool2.SetInputConnection(self.tool2.GetOutputPort())
        self.tool2Actor.GetProperty().SetColor(0.1, 0.2, 0.3)

        self.currentTransform = 0
        self.maxiterations = np.shape(transfromList)[0]
        self.scenPtCloud = VtkPointCloud()

        self.writer = writer


    def execute(self, iren, event):
        # self.writeVideo()
        self.showTools(iren)
        self.showPtCloud(iren)
        self.currentTransform += 1

    def writeVideo(self):
        self.writer.Start()
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(self.iren.GetRenderWindow())
        self.writer.SetInputConnection(windowToImageFilter.GetOutputPort())

        windowToImageFilter.SetInputBufferTypeToRGBA()
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()
        self.writer.Write()

    def showTools(self, iren):
        if self.currentTransform == self.maxiterations-1:
            iren.DestroyTimer(self.timerId)
            self.writer.End()
        tool1Transform = vtk.vtkTransform()
        tool1Transform.SetMatrix(self.mat2vtk(self.transformList[self.currentTransform][0]))
        tool2Transform = vtk.vtkTransform()
        tool2Transform.SetMatrix(self.mat2vtk(self.transformList[self.currentTransform][1]))
        self.tool1Actor.SetUserTransform(tool1Transform)
        self.tool2Actor.SetUserTransform(tool2Transform)
        self.tool1Actor.SetMapper(self.mapperTool1)
        self.tool2Actor.SetMapper(self.mapperTool2)

        print("iteration", self.currentTransform)
        iren.GetRenderWindow().Render()

    def showPtCloud(self, iren):
        if self.currentTransform == self.maxiterations-1:
            iren.DestroyTimer(self.timerId)
            self.writer.End()
        self.renderer.AddActor(self.scenPtCloud.vtkActor)
        self.scenPtCloud.clearPoints()
        for i in range(400):
            point = self.videoPtCloud[self.currentTransform, :, i]
            self.scenPtCloud.addPoint(point)
        iren.GetRenderWindow().Render()



    def mat2vtk(self, mat):
        obj = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                obj.SetElement(i, j, mat[i, j])
        return obj


class AddPointCloudTimerCallback:
    def __init__(self, renderer, iterations):
        self.iterations = iterations
        self.renderer = renderer

    def execute(self, iren, event):
        if self.iterations == 0:
            iren.DestroyTimer(self.timerId)
        pointCloud = VtkPointCloud()
        self.renderer.AddActor(pointCloud.vtkActor)
        pointCloud.clearPoints()
        for k in range(10000):
            point = 20*(np.random.rand(3)-0.5)
            pointCloud.addPoint(point)
        pointCloud.addPoint([0, 0, 0])
        pointCloud.addPoint([0, 0, 0])
        pointCloud.addPoint([0, 0, 0])
        pointCloud.addPoint([0, 0, 0])
        iren.GetRenderWindow().Render()
        if self.iterations == 30:
            self.renderer.ResetCamera()

        self.iterations -= 1

class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0
        self.actor1 = None
        self.actor2 = None
        self.transform1 = None
        self.transform2 = None

    def setActors(self, actor1, actor2):
        self.actor1 = actor1
        self.actor2 = actor2

    def execute(self, obj, event):
        print("vtkTimerCallback", self.timer_count)
        print("transform1", self.transform1)
        transforml = vtk.vtkTransform()
        transforml.SetMatrix(self.mat2vtk(self.transform1))
        self.actor1.SetUserTransform(transforml)

        print("transformr", self.transform2)
        transformr = vtk.vtkTransform()
        transformr.SetMatrix(self.mat2vtk(self.transform2))
        self.actor2.SetUserTransform(transformr)

        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1

    def mat2vtk(self, mat):
        obj = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                obj.SetElement(i, j, mat[i, j])
        return obj




