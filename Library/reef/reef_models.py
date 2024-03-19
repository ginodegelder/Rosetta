from processes import (ProfileZ, 
                       UniformGrid1D, 
                       ErosionConst, 
                       ErosOnly,
                       SeaLevel, 
                       VerticalDisp, 
                       WaterHeight, 
                       InitTopo,  
                       SLFile,
                       Construction,
                       MyHorizontalFactor, 
                       MyVerticalFactor, 
                       ErosiveMemory,
                       ErosiveMemory_ErosOnly,
                       SedimClastics,
                       InitTopoTerr,
                      )
import xsimlab as xs

#*********************************************************************
#*********************************************************************

reef = xs.Model({
    "vertical"  : VerticalDisp,
    "grid"      : UniformGrid1D,
    "water"     : WaterHeight,
    "profile"   : ProfileZ,
    "init"      : InitTopo,
    "sealevel"  : SeaLevel,
    "SLstory"   : SLFile,
    "construct" : Construction,
    "hfactor"   : MyHorizontalFactor,
    "vfactor"   : MyVerticalFactor,
    "eros"      : ErosionConst,
    "erosmem"   : ErosiveMemory,
    "depot"     : SedimClastics
})

#*********************************************************************

reef_platform = xs.Model({
    "vertical"  : VerticalDisp,
    "grid"      : UniformGrid1D,
    "water"     : WaterHeight,
    "profile"   : ProfileZ,
    "init"      : InitTopoTerr,
    "sealevel"  : SeaLevel,
    "SLstory"   : SLFile,
    "construct" : Construction,
    "hfactor"   : MyHorizontalFactor,
    "vfactor"   : MyVerticalFactor,
    "eros"      : ErosionConst,
    "erosmem"   : ErosiveMemory,
    "depot"     : SedimClastics
})

#*********************************************************************

reef_eros = xs.Model({
    "vertical"  : VerticalDisp,
    "grid"      : UniformGrid1D,
    "water"     : WaterHeight,
    "profile"   : ProfileZ,
    "init"      : InitTopo,
    "sealevel"  : SeaLevel,
    "SLstory"   : SLFile,
    "eros"      : ErosOnly,
    "erosmem"   : ErosiveMemory_ErosOnly,
#    "depot"     : SedimClastics
})
