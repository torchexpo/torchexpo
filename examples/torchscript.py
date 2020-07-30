"""torchexpo torchscript example"""
import torchexpo


fcn = torchexpo.vision.FCNResNet101()
fcn.extract_torchscript()
