from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2, numpy as np, spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from scipy.spatial.distance import euclidean

'''
Important Notes for each Part!!!

******* Part 1 *******

This part is simply creating AI model for emotion detection.
It uses CNN model to predict each emotion.
It is accuracy is approximately %55-60 percent of the emotion

I used FER-2013 emotion database for emotion detection AI
Link: https://www.kaggle.com/datasets/msambare/fer2013?resource=download

To start to this part, dataset must be present.

**********************
******* Part 2 *******

This part simply takes song information from spotify and saves it as csv file.
I created spotify developer account to achieve this cause.

I used Spotify's own playlists for each emotion:
Sad ID                             ------> 37i9dQZF1DX7qK8ma5wgG1?si=732c63912f384b52
Neutral (Calm) ID                  ------> 37i9dQZF1DXcy0AaElSqwE?si=74f2f25609924b65
Happy ID                           ------> 37i9dQZF1DXdPec7aLTmlC?si=4378d8923f944bd5
Horror (Fear, Suprise, Disgust) ID ------> 37i9dQZF1EIfgYPpPEriFK?si=357eb38ef4ca43e3
Angry ID                           ------> 37i9dQZF1EIgNZCaOGb0Mi?si=a114713b2e734b07

To start this part, client ID, client secret, track ID and save path must be given.

**********************
******* Part 3 *******

This part takes mean values of each column in csv file and prints it as list.

To start this part, csv file must be present and It's path must be given to the code.

**********************
******* Part 4 *******

This is the main part of the program.
It opens camera and waits user input (space).
When the user presses space, it takes photo, predicts each emotions percentage.
Then this percentage multiplied with the related values created in Part 3.
At the end of this multiplication, resulted list traversed at song database.
It find euclidean distance and takes the most similar 5 song and prints to screen.

For this part I used 1.2 million song database that I found from kaggle.
Link: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download

**********************
'''


part = input("\nWhich part do you want to (4 is suggested for last-user! Inputs needed for first 3 part!): \n\n"
             "1)Creating AI for emotion detection. \n"
             "2)Taking csv files from spotify playlist. \n"
             "3)Taking means value of each column of csv files. \n"
             "4)Using camera and database to create song suggestion.")


if part == '1':
    # Specify the path to your train and test datasets
    train_path = 'train'
    test_path = 'test'

    # Data preprocessing
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
    )

    # Building the CNN model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(train_generator, epochs=7, validation_data=test_generator)

    # Saving the trained model
    model.save('emotion_model.h5')

    # testing and printing the result of the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

elif part == '2':

    # Needed variables
    client_id = 'ENTER YOUR CLIENT ID'
    client_secret = 'ENTER YOUR CLIENT SECRET'
    track_id = 'ENTER THE TRACK ID'
    save_path = 'ENTER THE SAVE PATH'

    # Accessing spotify database
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Getting audio features of tracks
    audio_features = sp.audio_features([track_id])[0]

    # Saving them to the csv file
    df = pd.DataFrame([audio_features])
    df.to_csv(save_path, index=False)
    print("Data saved")

elif part == '3':

    other_file_path = 'ENTER YOUR PATH HERE'

    # Reading needed csv
    df_other = pd.read_csv(other_file_path)

    # Selecting only first 11 column
    df_other_selected = df_other.iloc[:, :11]

    # Calculating each columnns mean
    means = df_other_selected.mean()

    # Saving it to list
    mean_values = means.tolist()

    # Printing to mean on screen
    print("mean_values =", mean_values)

elif part == '4':

    # Opening the camera
    cap = cv2.VideoCapture(0)

    # Loading the previously made model
    emotion_model = load_model('emotion_model.h5')

    # Defining labels of the model
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    # Mean values created in previous step
    mean_values_angry = [0.5660000000000001, 0.81612, 4.78, -5.04438, 0.68, 0.136136, 0.068866858, 0.027984404999999993,
                         0.20739999999999995, 0.4914, 132.94598000000002]
    mean_values_happy = [0.704420, 0.732930, 5.080000, -5.391920, 0.660000, 0.085185, 0.120840, 0.003602, 0.157909,
                         0.622470, 121.296670]
    mean_values_fearful = [0.53374, 0.570164, 4.4, -9.63448, 0.42, 0.10407, 0.33629583999999996, 0.2700180076, 0.201926,
                           0.43048, 127.09622000000002]
    mean_values_sad = [0.503575, 0.373745, 4.6, -9.1181625, 0.875, 0.049756249999999995, 0.6663574999999999,
                       0.01028926275, 0.136375, 0.29327500000000006, 113.62836250000001]
    mean_values_neutral = [0.64546, 0.48525000000000007, 5.37, -9.51405, 0.55, 0.062011000000000004,
                           0.45286347000000005, 0.16932678590000005, 0.160989, 0.496, 113.98508]

    # Opening and selecting columns of 1.2m song database
    file_path = 'tracks_features.csv'
    df_selected_columns = pd.read_csv(file_path, usecols=range(9, 20))

    while True:
        # Capturing frame from webcam
        ret, frame = cap.read()

        # Displaying it while waiting input from the user
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)

        if key == ord(' '):

            # When the user press space, it preprocess captured image for model
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255.0

            # Saves emotion percantages here
            result = emotion_model(img)

            # Refactoring fear disgust and suprise
            emotion_percentages = {
                'Angry': result[0][0],
                'Happy': result[0][3],
                'Sad': result[0][4],
                'Neutral': result[0][6],
                'Disgust + Fear + Surprise': result[0][1] + result[0][2] + result[0][5]
            }
            # Prints values to the screen for user to see
            for emotion, percentage in emotion_percentages.items():
                print(f'{emotion}: {percentage * 100:.2f}%')

            # Calculating the weighted mean using emotion_percentages
            mean_values = [
                sum(emotion_percentages[emotion].numpy() * value for emotion, value in
                    zip(emotion_percentages.keys(), emotion_values))
                for emotion_values in zip(
                    mean_values_happy,
                    mean_values_sad,
                    mean_values_angry,
                    mean_values_neutral,
                    mean_values_fearful
                )
            ]

            # Calculating Euclidean distance between mean values and each row in the 1.2m database
            df_selected_columns['distance'] = df_selected_columns.apply(lambda row: euclidean(mean_values, row.values), axis=1)

            # Finding the best 5 row
            most_similar_indices = df_selected_columns['distance'].nsmallest(5).index

            # Reads the original csv file again and takes name(1), album(2), artists(4) columns
            df_original_columns = pd.read_csv(file_path, usecols=[1, 2, 4])

            # Prints these values to the screen
            print("\n\nBest songs for current emotion: \n")
            for index in most_similar_indices:
                print("Song Name:", df_original_columns.loc[index, 'name'])
                print("Album:", df_original_columns.loc[index, 'album'])
                print("Artists:", df_original_columns.loc[index, 'artists'])
                print("---")
            break
            # Breaks after one use