#!/usr/bin/env bash

# DESCRIPTION
# 
# Performs registration of abdominal MR images
# of the liver. The images are of either a T1w 
# or T2w contrast.
# 
# The primary goal is to register the T2w images
# to the T1w images (i.e. the T1w image is the 
# reference image, the T2w image is the moving 
# image). 
# 
# NOTE:
# 
# The T2w image(s) will be resampled to the T1w 
# image space.

# Load modules
module load mirtk/2.0.0

# Directory variables
scriptsdir=$(dirname $(realpath ${0}))
datadir=$(realpath ${scriptsdir}/../../../Data_TeamShare)
outdir=${datadir}/T2_to_T1

T1dir=$(realpath ${datadir}/T1*Liver*nifti)
T2dir=$(realpath ${datadir}/T2*Liver*nifti)


subs=( $(cd ${T1dir}; ls -d PT-*/ | sed "s@/@@g") )


for sub in ${subs[@]}; do

  echo "Processing: ${sub}"

  T1s=( $(ls ${T1dir}/${sub}/*.nii*) )
  T2s=( $(ls ${T2dir}/${sub}/*.nii*) )


  if [[ ${#T1s[@]} -eq 0 ]]; then
    # Check T1ws
    echo "${sub} does not have T1w image(s)"
    echo "${sub} does not have T1w image(s)" >> ${scriptsdir}/data.log
  elif [[ ${#T2s[@]} -eq 0 ]]; then
    # Check T2ws
    echo "${sub} does not have T2w image(s)"
    echo "${sub} does not have T2w image(s)" >> ${scriptsdir}/data.log
  else
    # Register images
    for T1 in ${T1s[@]}; do
      for T2 in ${T2s[@]}; do
        baseT1=$(basename ${T1%.nii*})
        baseT2=$(basename ${T2%.nii*})

        out_sub=${outdir}/${sub}
        out_img=${out_sub}/T2w_${baseT2}-to-T1w_${baseT1}.nii.gz

        if [[ ! -d ${out_sub} ]]; then
          mkdir -p ${out_sub}
        fi

        if [[ ! -f ${out_img} ]]; then
          bsub -M 10000 -W 800 -n 1 -R "span[hosts=1]" -J ${sub} \
          mirtk register -image ${T1} -image ${T2} -model Rigid -sim NMI -v -output ${out_img}
        fi
      done
    done
  fi

  # echo "T1: ${T1}"
  # echo "T2: ${T2}"
  # echo ""

done


# Sample code
# 
# mirtk register -image 044.nii.gz -image 072.nii.gz -model Rigid -sim NMI -v -output test.1.nii.gz