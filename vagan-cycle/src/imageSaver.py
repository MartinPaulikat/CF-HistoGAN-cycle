import tifffile as tiff
import numpy as np
from PIL import Image

#saves torch tensors as tiff or png at the experiment folder
class Saver:
    def saveAsPng(InputTensor, MapTensor, saveFolder, epoch, nameAddition, first):

        rgbInput = np.zeros((np.shape(InputTensor)[-2],np.shape(InputTensor)[-1],3), dtype=np.uint8)
        rgbInput[..., 0] = InputTensor[0, -3, ...] * 256
        rgbInput[..., 1] = InputTensor[0, -2, ...] * 256
        rgbInput[..., 2] = InputTensor[0, -1, ...] * 256

        imgIn = Image.fromarray(rgbInput)

        rgbMap = np.zeros((np.shape(InputTensor)[-2],np.shape(InputTensor)[-1],3), dtype=np.uint8)
        rgbMap[..., 0] = MapTensor[0, -3, ...] * 256
        rgbMap[..., 1] = MapTensor[0, -2, ...] * 256
        rgbMap[..., 2] = MapTensor[0, -1, ...] * 256

        imgMap = Image.fromarray(rgbMap)
        imgOut = Image.fromarray(rgbInput + rgbMap)

        #save the input only if it is the first call of this function
        if first:
            path = saveFolder + '/input_samplesHE_' + nameAddition + '.png'
            imgIn.save(path)
        path = saveFolder + '/map_samplesHE_' + nameAddition + '_' + str(epoch) + '.png'
        imgMap.save(path)
        path = saveFolder + '/output_samplesHE_' + nameAddition + '_' + str(epoch) + '.png'
        imgOut.save(path)

    def saveAsTiff(InputTensor, MapTensor, saveFolder, epoch, nameAddition, first):

        imgIn = np.array(InputTensor)
        imgMap = np.array(MapTensor)
        imgOut = imgIn + imgMap

        #save the input only if it is the first call of this function
        if first:
            path = saveFolder + '/input_samples_' + nameAddition + '.tif'
            tiff.imsave(path, imgIn)
        path = saveFolder + '/map_samples_' + nameAddition + '_' + str(epoch) + '.tif'
        tiff.imsave(path, imgMap)
        path = saveFolder + '/output_samples_' + nameAddition + '_' + str(epoch) + '.tif'
        tiff.imsave(path, imgOut)

    def saveSanity(sanity, saveFolder, epoch, nameAddition):

        path = saveFolder + '/map_samples_' + nameAddition + '_' + str(epoch) + '.tif'
        tiff.imsave(path, sanity)

        rgbSanity = np.zeros((np.shape(sanity)[-2],np.shape(sanity)[-1],3), dtype=np.uint8)
        rgbSanity[..., 0] = sanity[0, -3, ...] * 256
        rgbSanity[..., 1] = sanity[0, -2, ...] * 256
        rgbSanity[..., 2] = sanity[0, -1, ...] * 256

        imgSanity = Image.fromarray(rgbSanity)

        path = saveFolder + '/sanity_samplesHE_' + nameAddition + '_' + str(epoch) + '.png'
        imgSanity.save(path)