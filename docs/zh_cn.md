# <center> FlashFace 使用注意事项

与之前工作相比，FlashFace 的最大特点是可以通过语言对脸部实现精准的控制（控制年龄，性别，饰品，表情等等），同时生成图片不是照搬参考图中的人脸，而是有一定的 varience，但是这也导致其使用方式与之前工作不同，我们将用户反馈的的例子在此列举，并提供解决方案，以期待能帮助更多的人

## 1. prompt 中使用  “A man”  或者 “A woman” 得到较差的人脸相似度
[issues-11](https://github.com/ali-vilab/FlashFace/issues/11) 以及 [twitter](https://x.com/askerlee/status/1782628828164305305) 有用户 report 了这类问题. 

因为 FlashFace 中语言是可以强影响人脸的，比如提供一个一个年轻人的人脸，他可能会生成一个有胡子的中年人形象，仅仅提供 “A man” 这样模糊的语言描述往往不能得到令人满意的图片，因此你需要在 prompt 提供较多的细节，比如年纪，比如 “A handsome young man”,“A beautiful young woman”.

另一方面，类似 InstantID 使用关键点得到了稳定的 ID 保持能力，FlashFace 也提供了一个更弱的人脸控制信号``face_bounding_box``来控制构图以及提升 ID 保持效果(关键点会失去过多人脸 varience) ，用户可以使用类似 [0.3, 0.1, 0.6, 0.4] 这样的中间人脸位置 ，来提升人脸相似度.

此外，我们也提供了 ``lamda_feature``， ``face_guidance``， ``step_to_launch_face_guidance``来控制人脸相似度，lamda_feature : 0.8 ~ 1.3，表示使用 reference feature 的强度，一般仅仅调节此参数就可以得到满意的图片， 进一步调高 ``face_guidance``， ``step_to_launch_face_guidance`` 则可以帮助您保留更多的面部细节.

 ## 2. 亚洲人相似度较低
 FlashFace 训练集中包含非常少的亚洲人物，导致对于亚洲人脸定制需要一些额外的参数调整，这或许也部分导致了[issues-15](https://github.com/ali-vilab/FlashFace/issues/15), [issues-11](https://github.com/ali-vilab/FlashFace/issues/11) 以及 [twitter](https://x.com/askerlee/status/1782628828164305305) 


 如果定制的亚洲人脸相似度不够，您可以尝试在 prompt 中添加 Asian 如“A handsome young Asian man”,“A beautiful young Asian woman”. 同时增加 ``face_bounding_box``= [0.3, 0.1, 0.6, 0.4] 之类的人脸位置 ，亚洲人物我们也建议使用更高的``lamda_feature``， ``face_guidance``， ``step_to_launch_face_guidance``, 例如[issues-15](https://github.com/ali-vilab/FlashFace/issues/15) 中我使用的
 
 ```python
 face_bbox =[0.3, 0.2, 0.6, 0.5] 
# bigger these three parameters leads to more fidelity but less diversity 
lamda_feat = 1.2
face_guidence = 3
step_to_launch_face_guidence = 800
```

# 总结

当您生成的图片不满意，特别是 ID 相似度不够，你可以通过以下建议进行改进

1. 让你的 positive prompt 更加细致，不要使用过于模糊的 “A man/woman”  ，同时使用negtive prompt 去除不需要的特征
  
   因为 FlashFace 的最大特点就是 language 可以非常好的控制面部特征，当你的prompt 过于模糊，它生成的可能是其他年龄或者包含其他面部特征的人脸，例如 A man ，但是参考图是一个年轻人，可能会生成一个脸部发腮并且伴有胡子的中年人，因此您最好指定生成人物的大致年纪，比如 Handsome young / beautiful young 会显著改善这一状况，这样的形容词可以在保持 ID 的前提下让你的脸更加吸引人，同时，如果生成人物存在你不需要的面部特征，如胡子，皱纹，您可以在 negative prompt 中添加 “beard, wrinkle” 来去除这些特征, 同时对于亚洲人物，可以尝试在 prompt 添加 ``Asian``. 可以参考 [issues-11](https://github.com/ali-vilab/FlashFace/issues/11) 查看一些参数经验.

2. 增加 Face Position 
  可以添加一个 Face position，比如 [0.3, 0.1, 0.6, 0.4]，表示人脸大概在图片的中间，可以避免生成一些过大或者过小的人脸，可以大大提升生成效果, 但是会牺牲一些人脸的多样性.


3. 调高 lamda_feature， face_guidance， step_to_launch_face_guidance
  
    ``lamda_feature`` : 0.8 ~ 1.3，使用 reference feature 的强度，一般仅仅调节此参数就可以得到满意的图片 
    ``face_guidance``， ``step_to_launch_face_guidance``，调节这两个参数可以获得更多的脸部细节的保留，可以参考ipynb 最后一个例子中参数的设置，来学习如何保留面部细节.

    同时对于亚洲人物，我们建议调高这三个参数，比如调整到.
    ```
    ```python
    face_bbox =[0.3, 0.2, 0.6, 0.5] 
    # bigger these three parameters leads to more fidelity but less diversity 
    lamda_feat = 1.2
    face_guidence = 3
    step_to_launch_face_guidence = 800
    ```
    但是有时这三个值过大会导致脸部出现贴图感甚至 artifacts, 如有类似情况请对应调小 ``lamda_feat`` 与 ``face_guidence``