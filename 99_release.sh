#!/bin/bash -f
ROOT="face_detector"
MODEL="onnx_model"
LIB="detlib_onnx"

if [[ -d "${ROOT}" ]]
then
    echo "remove exist directory : ${ROOT}"
    rm -rf ${ROOT}
fi

mkdir ${ROOT} 
mkdir ${ROOT}/${LIB}
mkdir ${ROOT}/${LIB}/"detector"

cp -r ${MODEL} ${ROOT}
cp    ${LIB}/"__init__.py"              ${ROOT}/${LIB}/
cp    ${LIB}/"detector"/"__init__.py"   ${ROOT}/${LIB}/"detector"/ 
cp    ${LIB}/"detector"/"detect.py"     ${ROOT}/${LIB}/"detector"/
cp    ${LIB}/"detector"/"nms.py"        ${ROOT}/${LIB}/"detector"/
cp    ${LIB}/"detector"/"utils.py"      ${ROOT}/${LIB}/"detector"/
cp    ${LIB}/"detector"/"vision.py"     ${ROOT}/${LIB}/"detector"/

cp -r "demo_onnx.py" ${ROOT}
