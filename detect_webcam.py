import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(r"C:\Users\arote\OneDrive\Desktop\Face-Detection\haarcascade\haarcascade_frontalface_default.xml")

# Check XML loaded correctly
if face_cascade.empty():
    print("Error loading Haar Cascade XML file")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw rectangles
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show output
    cv2.imshow(
        'Real-Time Face Detection',
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()