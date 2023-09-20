# Go to your 3D slicer path, and run Slicer.exe --python-script "file_path/run_slicer.py"
import DICOMLib
import re
import os

dicomDataDir_all = 'D:/QC_data_First_step/NYU/batch2/QC_DWI/DWI_b800/'

lst = sorted(os.listdir(dicomDataDir_all))


for subDataDir in lst:
    
    dicomDataDir = dicomDataDir_all + subDataDir + '/' + os.listdir(dicomDataDir_all + subDataDir)[0]
    outputDir =  dicomDataDir_all + subDataDir
    
    # Import the DICOM data from the current directory
    DICOMLib.importDicom(dicomDataDir)
    
    # Get a list of all DICOM files in the current directory
    dicomFiles = slicer.util.getFilesInDirectory(dicomDataDir)
    
    # Get loadable objects from the DICOM files
    loadablesByPlugin, loadEnabled = DICOMLib.getLoadablesFromFileLists([dicomFiles])
    
    # Load the loadable objects and get their node IDs
    loadedNodeIDs = DICOMLib.loadLoadables(loadablesByPlugin)
    
    # Loop through each loaded node ID
    for loadedNodeID in loadedNodeIDs:
        # Get the node object using its ID
        node = slicer.mrmlScene.GetNodeByID(loadedNodeID)
        
        # Try to sanitize the node name to create a safe file name
        try:
            safeFileName = re.sub(r'(?u)[^-\w.]', '', node.GetName().strip().replace(' ', '_'))
        except AttributeError:
            pass
        
        # Try to save the node to a .nrrd file format in the output directory
        try:
            slicer.util.saveNode(node, '{0}/{1}.nrrd'.format(outputDir, safeFileName))
        except AttributeError:
            pass
