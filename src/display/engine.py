from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_Color
from OCC.Display.SimpleGui import init_display
from OCC.Display.OCCViewer import Viewer3d

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional

from src.display.scene import Scene


class DisplayEngine(ABC):
    def __init__(self) -> None:
        self.scenes: List[Scene] = []
        
    def add_scene(self, scene: Scene) -> None:  
        self.scenes.append(scene)
        return  
    
    def remove_scene(self, scene: Scene) -> None:   
        self.scenes.remove(scene)
        return
    
    def clear(self) -> None:
        self.scenes.clear()
        return  
    
    @abstractmethod 
    def display(self) -> None:
        pass    
    
class SimpleDisplayEngine(DisplayEngine):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def _generate_display_func(self, scene: Scene, display: Viewer3d) -> None:
        def display_scene():
            display.EraseAll()
            display.View_Iso()
            
            if scene.view == 'z':
                display.View.SetProj(0, 0, 1)
            elif scene.view == 'x':
                display.View.SetProj(1, 0, 0)
            
            for entity in scene:
                if entity.brep_solid == None:
                    continue
                display.DisplayShape(
                    shapes=entity.brep_solid,
                    color=entity.color, 
                    transparency=entity.transparency
                )
                if entity.msg:
                    display.DisplayMessage(
                        entity.center_pnt, 
                        entity.msg, 
                        update=True, 
                        height=30
                    )
            display.FitAll()
        # 함수의 이름을 scene 이름으로 설정
        display_scene.__name__ = scene.name
        return display_scene

    def display(self) -> None:
        display, start_display, add_menu, add_function_to_menu = init_display() 
        
        display.View.SetBgGradientColors(
            Quantity_Color(Quantity_NOC_WHITE),
            Quantity_Color(Quantity_NOC_WHITE),
            2,
            True)
        
        # scene 메뉴 추가
        add_menu("scenes")
        for scene in self.scenes:
            add_function_to_menu(
                "scenes",  # 메뉴 항목 이름
                self._generate_display_func(scene, display))
        start_display()
        return
    
    
if __name__ == '__main__':
    scene1 = Scene("")