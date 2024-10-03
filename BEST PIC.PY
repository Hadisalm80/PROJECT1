import cv2
import numpy as np

image = cv2.imread('equalized_reduced_cropped_9003658L.png')
mask = cv2.imread('cropped_mask_output.png', 0)

img_height, img_width, _ = image.shape
mask_height, mask_width = mask.shape

best_match = None
best_match_score = float('inf')
for start_y in range(img_height - mask_height + 1):
    for start_x in range(img_width - mask_width + 1):
        roi = image[start_y:start_y + mask_height, start_x:start_x + mask_width]

        score = np.sum(roi == mask[:, :, np.newaxis])

        if score < best_match_score:
            best_match_score = score
            best_match = (start_x, start_y, roi)

if best_match is not None:
    start_x, start_y, best_roi = best_match
    final_result = np.zeros_like(image)

    final_result[start_y:start_y + mask_height, start_x:start_x + mask_width] = best_roi

    cv2.imwrite('best_match_result.jpg', final_result)

    cv2.imshow('Best Match Result', final_result)
    cv2.waitKey(0)
else:
    print("No match found.")

cv2.destroyAllWindows()
