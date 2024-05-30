# <center> Precautions for Using FlashFace

Compared to previous approaches, FlashFace's distinctive feature is granting precise control over facial features through language (including age, gender, accessories, expressions etc.), while ensuring that the generated image isn't a direct replicate of the face from the reference picture but rather possesses a certain degree of variance. However, this also makes its usage differ from preceding works. We've gathered and presented user feedback and examples here, along with proposed solutions, with the hope that it may aid a broader user base.

## 1. Poor facial similarity when using "A man" or "A woman" in prompts
This type of issue was reported by users in [issues-11](https://github.com/ali-vilab/FlashFace/issues/11) and on [Twitter](https://x.com/askerlee/status/1782628828164305305).

In FlashFace, language can highly influence the generated face images. For example, when supplied a young face as a reference, the model might generate an image of a middle-aged man with a beard. Therefore, a vague description such as "A man" often cannot yield satisfactory images. It is recommended to provide more details in the prompt, such as "A handsome young man" or "A beautiful young woman".

On the other hand, just as InstantID maintains stable identity through the use of face key points, FlashFace also provides a weaker facial control signal, ``face_bounding_box``, to control the composition and improve identity preservation. Users can use a midpoint face position like [0.3, 0.1, 0.6, 0.4] to enhance facial similarity.

Additionally, we provide other parameters like `lamda_feature`, `face_guidance`, and `step_to_launch_face_guidance` for further control:
- `lamda_feature` (ranging from 0.8 to 1.3) represents the intensity of usage of the reference feature. Normally, adjusting this parameter alone can produce satisfactory images.
- Further increasing `face_guidance` and `step_to_launch_face_guidance` can help preserve more facial details.

## 2. Lower similarity for Asian faces
The FlashFace training dataset contains few Asian individuals, leading to the need for additional parameter adjustment for Asian faces. This might have contributed to the issues raised in [issues-15](https://github.com/ali-vilab/FlashFace/issues/15), [issues-11](https://github.com/ali-vilab/FlashFace/issues/11) and on [Twitter](https://x.com/askerlee/status/1782628828164305305).

If the similarity for specified Asian faces isn't sufficient, you could try adding "Asian" in the prompt, like “A handsome young Asian man” or “A beautiful young Asian woman”. Simultaneously, you can add a ``face_bounding_box``, for example ``[0.3, 0.1, 0.6, 0.4]``. For better similarity of Asian faces, we also suggest employing higher values for ``lamda_feature``, ``face_guidance``, and ``step_to_launch_face_guidance``. As an example, in [issues-15](https://github.com/ali-vilab/FlashFace/issues/15), I used the following:

```python
face_bbox =[0.3, 0.2, 0.6, 0.5] 
# Increasing these three parameters leads to more fidelity but less diversity 
lamda_feat = 1.2
face_guidence = 3
step_to_launch_face_guidence = 800
```


# Conclusion

If the generated image does not meet your expectations, especially in terms of ID similarity, you can consider the following suggestions for improvement:

1. Make your positive prompts more specific and avoid using vague descriptions such as "A man/woman". At the same time, use negative prompts to remove unnecessary features.

   One of the main features of FlashFace is its ability to control facial features through language. However, if your prompt is too vague, it may generate a face with unexpected age or characteristics. For instance, if a "A man" is used as a prompt with a reference image of a young individual, it may generate the face of a middle-aged man with a beard. In this case, specifying the approximate age of the person to be generated, such as "Handsome young" or "Beautiful young", can significantly improve the results. These descriptors allow you to make the face more attractive while preserving the ID. Additionally, if the generated face includes features you do not want, such as beards or wrinkles, you can add "beard, wrinkle" to the negative prompts to eliminate these features. For Asian people, you might consider adding "Asian" to the prompt. You can refer to [issues-11](https://github.com/ali-vilab/FlashFace/issues/11) for some parameter experience.

2. Add a face position.
   
   For example, specifying a face position such as [0.3, 0.1, 0.6, 0.4], which positions the face roughly in the middle of the image, can prevent the generation of excessively large or small faces and thus significantly improve the image result. However, it may sacrifice some facial diversity.

3. Increase `lamda_feature`, `face_guidance`, and `step_to_launch_face_guidance`.

   `lamda_feature` (ranging from 0.8 to 1.3) signifies the intensity of using the reference feature. Usually, adjusting this parameter alone can generate satisfactory images. 
  
   Adjusting `face_guidance` and `step_to_launch_face_guidance` can help retain more facial details. Refer to the settings of the parameters in the last example in the ipynb file to learn how to preserve facial details.
   
   For Asian faces, we recommend increasing these three parameters as follows,
   
   ```python
   face_bbox =[0.3, 0.2, 0.6, 0.5] 
   # Increasing these three parameters leads to more fidelity but less diversity 
   lamda_feat = 1.2
   face_guidence = 3
   step_to_launch_face_guidence = 800
   ```
   This adjustment can significantly enhance the representation of Asian faces. However, sometimes if these three values are too high, it will cause the face to appear textured or even artifacts. If there is a similar situation, please adjust ``lamda_feat`` and ``face_guidence`` to smaller values.
