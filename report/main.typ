#let conf(title: str, subtitle: str, date: str, authors: (), doc) = {
  page[#{
    show heading.where(level: 1): it => block(width: 100%)[
      #set align(center)
      #set text(1.5em, weight: "regular") 
      #block[#it.body]
    ]
    show heading.where(level: 2): it => block(width: 100%)[
      #set align(center)
      #set text(1.25em, weight: "extralight") 
      #block[#it.body]
    ]
    
    image("rmit.svg", width: 25%)
    
    v(1fr)

    align(center + horizon)[#{
      heading(level: 1, title)
      heading(level: 2, subtitle)
      v(0.5em)
      par(justify: false, date)          
    }]

    v(1fr)

    block[
      *Group*: S3_G5 \
      *Members*:
      #list(..authors.map(author => [
        #author.name - #link("mailto:" + author.email)
      ]))     
    ]
  }]

  page(numbering: "1")[
    #{
      set align(left)
      set text(11pt, hyphenate: false)
      set heading(numbering: "1.")
      set par(justify: true)
      show table.cell.where(y: 0): strong
      set table(
        stroke: (x, y) => if y == 0 {
          (bottom: 0.7pt + black)
        },
        align: (x, y) => (
          if x > 0 { center }
          else { left }
        )
      )

      counter(page).update(1)
      
      doc
    }
  ]
}

#show: doc => conf(
  title: [COSC2753 Machine Learning - Assignment 2],
  subtitle: [Group Report],
  date: [May 16, 2025],
  authors: (
    (
      name: "Nguyen Vu Trong Nhan",
      email: "s4028019@rmit.edu.vn"
    ),
    (
      name: "Le Duy Quang",
      email: "s3912105@rmit.edu.vn"
    ),
    (
      name: "Do Khoa Nguyen",
      email: "s3978796@rmit.edu.vn"
    ),
    (
      name: "Duong Hoang Anh Khoa",
      email: "s3864111@rmit.edu.vn"
    )
  ),
  doc,
)

= Exploratory Data Analysis

== Metadata and Images Validation

Here, metadata and images are validated to ensure they have the correct number of relations in-between. Missing values in metadata are checked to be *non-empty*, so imputation is not needed.

== Statistical Overview

There are *10 unique `label` types* and *10 `variety` types*, with `age` being in range of 45 to 82. With that, this data could be categorized into *categorical* and *discrete* feature types, where `variety` and `label` being limited - thus *categorial*, and `age` being *discrete* as it is countable.

=== Data Shape

Visualize the dataset with distribution of each features in @distribution, observations could be seen:
- `age`: fairly Gaussian-like distribution, no outliers, and a suitable inter-quartile range.
- `disease`: about most dataset has high contribution, while other disease contribute less than half.
- `variety`: one feature (`ADT45`) is the majority gaining 67% contribution of all plant types.

#figure(
  image("distribution.svg", width: 80%),
  caption: [Distribution of `age`, `disease` and `variety` in dataset.],
) <distribution>

=== Relationships between Features

Factorize categorial values and plot correlation heat-map, features seems to correlate quite poorly.

When compare `age` to disease `label`, unhealthy disease tend to has larger inter-quartile range, while healthy `label` has small variation and small amount of outliers could be seen. Compare `age` to `variety`, `ADT45` has more distributed data while some variety are found to has only one `age` value, and others has more "controlled" values.

Finally, the contribution of data-points of `ADT45` to the dataset, most of data comes from only one plant `variety`, and its distribution could be spot as the majority. By that, further resampling method might be used to improve the performance of trained models later on.

== Image Analysis

=== Metadata and Duplicates

Common issues on images are analyzed with CleanVision @cleanvision to ensure they have standard lighting, identical aspect-ratio and image size, and image duplicates. Here, it is found to have identical aspect ratio and size, however consists of around 400 duplicates. Moreover, some of the duplicate set are found to consists of different metadata.

Removing the duplicates and unmatched metadata at this stage saw a minor dataset reduction of 2%.

=== Color Featuring

Analyzing mean color distributions as shown in @color-intensity, in higher color intensity ($x > 150$), the green color-space is the majority due to the fact that images are capture of plants @plant-histogram.

#figure(
  image("color-distribution.svg", width: 80%),
  caption: [Mean color distribution of non-duplicated images in dataset],
) <color-intensity>

Sample some images that has low green intensity, due to environmental color changes or capturing with incorrect white balance setting resulting in color has more significant blue hues @white-balance; experiment could be conduct to see if model performance would improve with some of the images removed.

#figure(
  image("nipy.svg", width: 80%),
  caption: [Image samples after applying `nipy` color-map]
)

Flatten out the color-space and apply `nipy` color-map @nipype, the intensity of the red channel could be highlight and seen as a differentiate factor that could plays a major role in training our model.

Experimenting with a single color-channel disabled @image-processing, information about the image could be gain in small edges, especially `brown_spot` where clear distinction of colors near the edge could be seen.

== Handling the Imbalance

As `label` and `variety` are imbalanced, SMOTE @smote is performed to over-sample the dataset @imbalancelearn.

Here, over-sampling is chosen due to the large gap in both features, where under-sampling would remove many samples of the dataset and might become a factor in training. Moreover, the decision to use SMOTE over ADASYN @adasyn is due to overall performance in many researches @smote-adasyn-review @smote-bsmote-adasyn-study.

= Training

== Task 1. Paddy Multi-class Label Classification

=== Data Preparation
We perform training with image at original size of `640x480` and square `256x256` size, to assess the impact of spatial dimensions on classification accuracy. For image color channel diversity, we tested images with standard RGB, red-removed (G+B), green-removed (R+B), blue-removed (R+G), and spectral projection. We evaluate the original train images with the variant of colors, while generate an addition non-imbalance SMOTE @smote  dataset taken from the train validation split to handle up-sampling.  

=== Model Architecture and Justification

ResNet @resnet18, AlexNet @li2022improved, ConvNext @rohman2024classification and MobileNetV3 @mobilenet were tested across various color channels. Standard RGB images results were at least 10% better than the remaining channel in the majority of model. While MobileNetV3 showed signs of overhitting, ConvNeXT was resource-intensive with limited benefits. On the other hand, AlexNet and ResNet offered solid accuracy with minimal over-fitting, making them the most suitable option.

From these two, we enhanced AlexNet with Inception-style parallel branches. As a result, accuracy was improved by 7%, peaking at 90% with only 0.2 in loss. Furthermore, adding a Squeeze-and-Excitation (SE) @hu2018squeeze block further refined feature attention, stabilized training after initial fluctuations and achieved 93.953% validation accuracy with reduced over-fitting.

Finally, Grad-CAM @selvaraju2016grad was applied to guide feature selection and select augmentation strategy. Despite slightly drop in accuracy, using coarse dropout from Albumentations @buslaev2020albumentations method helps improve real usage scenario generalization. Additionally, we experimented with SMOTE-based oversampling, but it significantly reduced the model's accuracy and increased over-fitting, suggesting limited effectiveness in this context.

=== Performance Evaluation

After 50 epochs, the model achieved strong performance with a validation accuracy of 91.52% and training accuracy of 90.96%, indicating good generalization. The close loss values (training: 0.2733, validation: 0.2866) further confirm stable learning. We observe that the all the weight average from precision, recall, and f1-score all reach 0.92.

#figure(
  image("task1_accuracy_loss_plot.svg", width: 70%),
  caption: [Epoch Accuracy],
)

=== Ultimate Judgement

Testing with different models with different color channels at the beginning help understand the model strength as well as the effectiveness of that color channel among the tested models. Thus, the choice of AlexNet become more robust for furthure examination and development. The final model in combination of AlexNet, Inception technique, and SE block has significantly increase compared to the orignial AlexNet model by up to 7% in validation accuracy and achieve better generalization. 

== Task 2. Paddy Variety Classification using 4-Stack Color CNN

=== Data Preparation

At this stage, images were resized into `128x128`, and categorical features was numerically encoded.

=== Model Architecture and Justification

MobilenetV2, VGG16, and a custom-built Convolutional Neural Network (CNN) were implemented for model selection phase. Both `MobileNetV2` @keras_mobilenetv2 and `VGG16` @keras_vgg16 shared limitations in performance due to their capability of only accepting only 3-channel `RGB` inputs. While these two pretrained models offered strong transfer learning capabilities @Adhinata2021, they were not capable of leveraging the additional spectral channel (`nipy_spectral`) that are provided by our data, they tended to underperform, particularly on minority classes.

In contrast, CNN was designed to support 4-channel input @Gopalapillai2021-on, in this case stacking RGB and spectral images allowing it to extract richer features across visual and spectral domains. Hence, The 4-stack color CNN was selected as the most suitable architecture for our data.

=== Performance Evaluation

After training for 27 epochs which is due to early stopping implemented, the final result were:

#align(center)[
  `loss: 0.1358 - accuracy: 0.9524 - val_loss: 0.2974 - val_accuracy: 0.9289`
]

The small gap between 95.2% of training accuracy and 92.9% of validation accuracy indicates that the model generalizes well without significant over-fitting. The moderate validation loss (0.2974) suggests that the model assigning high probability to the correct class, aligning with expected performance for moderate-complexity CNNs trained on limited-size datasets without pretraining.

#figure(
  image("epoch-accuracy.svg", width: 50%),
  caption: [Epoch Accuracy],
)

=== Ultimate Judgement

The training history, confusion matrix and the classification report were used to evaluate the 4-channel CNN Model. Training and validation accuracy steadily improved, reaching 95.7% and 94% respectively, with minimal divergence, indicating good generalization and effective regularization via data augmentation and early stopping. The confusion matrix shows strong dominant classes in ADT45 and IR20, on the other hand, likely due to visually similarity, minor confusion were shown in class like Ponni and RR, confirmed with the classification report's macros F1-score of 0.87 and weighted average of 0.94, reflecting balanced performance across all classes.

From this, we conclude that the model is well-generalized, handles class imbalance robustly, and is suitable for practical deployment in rice variety classification tasks. However, the model's performance on classes with very few samples such as Surya and RR still lags slightly, indicating a limitation in data availability rather than the architecture's capability. Nevertheless, this custom 4-channel CNN is the most effective model for our problem space considered on the accuracy, class balance, training stability and the efficiency of the model.

== Task 3. Age Estimation using EfficientNetB0

=== Data Preparation

The pipeline loads metadata from `meta_train.csv` and collects valid (`image_path`, `age`) pairs from spectral images (`nipy_spectral.jpg`) stored in labeled subdirectories. Only images with matching metadata are included for training.

=== Model Architecture and Justification

InceptionV3 showed high memory usage and slow convergence with minimal performance gains for age regression. We switched to EfficientNetB0 for its lightweight design, faster processing, and lower resource consumption, making it a more practical and effective choice.

We use a convolutional neural network based on EfficientNetB0 as the backbone architecture. This model was chosen for its high performance-to-parameter ratio and its efficient scaling of depth, width, and resolution @Tan2019-lw. The EfficientNetB0 model is used without pretrained weights (weights=None) to allow full training on our domain-specific dataset.

*Architecture Overview* Input Layer: 128×128×3, EfficientNetB0 Base: Feature extractor without top classification layers, Two dense layers (256 & 128 units, ReLU) with L2 regularization, Output Layer: Single unit (linear activation) for age regression

This strategy gradually reduces the learning rate during training to avoid overshooting minima and helps with model convergence @Goodfellow-et-al-2016.

The model was trained using the *Adam* optimizer and *Mean Absolute Error (MAE)* as both the loss function and evaluation metric, which is suitable for continuous regression tasks like age prediction.

=== Performance Evaluation

After training for 60 epochs, the final results are:
- Training MAE: 3.78
- Validation MAE: 5.53

The low difference between training and validation MAE indicates the model generalizes relatively well. This level of MAE is consistent with baseline performance in age estimation tasks from facial or biological imagery when not using pretrained weights or large datasets.

#figure(
  image("epoch-accuracy-spec-task3.svg", width: 50%),
  caption: [Epoch logs showing steady decrease in MAE and loss.],
)

=== Ultimate Judgement

The model demonstrates strong potential for age prediction from spectral images. Achieving a validation MAE of 5.53 is promising, especially given that:
- No pretrained weights were used,
- The image resolution was relatively low (128×128),
- Only basic preprocessing was applied.

These results suggest that the current pipeline provides a solid foundation for further improvement. With fine-tuning, transfer learning, and additional data or augmentation, the model could potentially reach a lower MAE, which is competitive in age regression tasks.

#pagebreak()

#heading(numbering: none)[Appendix]

#figure(
  image("application.jpg", width: 30%),
  caption: [Single-page application for onsite model prediction]
)

#let results = csv("COSC2753_A2_S3_G5.csv")
#let predictions_count = 20

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    [*`image_id`*], [*`label`*], [*`variety`*], [*`age`*],
    ..results.slice(1, 20).flatten(),
  ),
  caption: [First #predictions_count results of model predictions]
)

#pagebreak()

#bibliography("references.bib", title: "References", style: "ieee")