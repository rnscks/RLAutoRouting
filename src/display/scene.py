from OCC.Display.SimpleGui import init_display    
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_Color
from OCC.Core.V3d import V3d_Zpos
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs   
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing   
from OCC.Core.IFSelect import IFSelect_RetDone  

from typing import List, Optional
from PIL import Image
import os
import tempfile

from src.display.entity import Entity    


class Scene:
    def __init__(self, name: str = "") -> None:
        self.entities: List[Entity] = []
        self.name: str = name   
        self.view: str = ''
    
    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)
        return
    
    def add_entities(self, entities: List[Entity]) -> None: 
        self.entities.extend(entities)
        return  
    
    def remove_entity(self, entity: Entity) -> None:
        self.entities.remove(entity)
        return
    
    def set_view(self, view: str = 'z'):
        self.view = view
        return
    
    def display(self) -> None:
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        display.View.SetBgGradientColors(
            Quantity_Color(Quantity_NOC_WHITE),
            Quantity_Color(Quantity_NOC_WHITE),
            2,
            True)
        if self.view == 'z':
            display.View.SetProj(0, 0, 1)  # Z축 방향에서 바라보기
        elif self.view == 'x':
            display.View.SetProj(1, 0, 0)
        
        for entity in self.entities:
            if isinstance(entity.brep_solid, type(None)):
                continue
            display.DisplayShape(entity.brep_solid, update=True, color=entity.color, transparency=entity.transparency)  
            if entity.msg != "":
                display.DisplayMessage(entity.center_pnt, entity.msg, height=22)
        
        start_display() 
        return  
    
    def capture(self, file_name: str) -> None:
        display, _, _, _ = init_display()           
        display.View.SetBgGradientColors(
            Quantity_Color(Quantity_NOC_WHITE),
            Quantity_Color(Quantity_NOC_WHITE),
            2,
            True)
        display.FitAll()
        display.View.SetProj(V3d_Zpos) 
        
        for entity in self.entities:
            if isinstance(entity.brep_solid, type(None)):
                continue
            display.DisplayShape(entity.brep_solid, update=True, color=entity.color, transparency=entity.transparency)  
            if entity.msg != "":
                display.DisplayMessage(entity.center_pnt, entity.msg)
        display.View.Dump(file_name)
        return
    
    def save_to_stp(self, file_name: str) -> None:
        try:
            compounds = [entity.brep_solid for entity in self.entities if entity.brep_solid is not None]  
            compound_builder = BRepBuilderAPI_Sewing()
            for compound in compounds:  
                if compound is None or compound.IsNull():
                    continue
                compound_builder.Add(compound)
            
            compound_builder.Perform()
            compound = compound_builder.SewedShape()

            stp_writer = STEPControl_Writer()
            stp_writer.Transfer(compound, STEPControl_AsIs)

            status = stp_writer.Write(file_name + ".stp")   

            if status == IFSelect_RetDone:
                print(f"STP 파일이 성공적으로 저장되었습니다: {file_name}.stp")
            else:
                print("STP 파일 저장 중 오류가 발생하였습니다.")
        except:
            raise ValueError('Fail to save the file')   
        return 
    
    def __iter__(self):
        return iter(self.entities)
    