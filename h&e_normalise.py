import os
import cv2 
import numpy as np
def normalise(true_dir, false):
    Io = 240  # Normalizing factor for intensity
    alpha = 1
    beta = 0.15

    i = 0
    for directory in [true_dir]:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            tile = cv2.imread(filepath)
            if tile is None:
                print(f"Skipping {filename}, not a valid image.")
                continue

            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            ######## Step 1: Convert RGB to OD ###################
            HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]])
            maxCRef = np.array([1.9705, 1.0308])
            h, w, c = rgb.shape
            rgb = rgb.reshape(-1, 3)
            OD = -np.log10((rgb.astype(float) + 1) / Io)

            ######## Step 2: Remove data with OD intensity less than β ########
            ODhat = OD[~np.any(OD < beta, axis=1)]  # Filter out transparent pixels

            ######## Step 3: Calculate SVD on OD tuples ######################
            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

            ######## Step 4: Create plane from SVD directions #################
            That = ODhat.dot(eigvecs[:, 1:3])

            ######## Step 5 & 6: Project onto plane, normalize, and calculate angle ########
            phi = np.arctan2(That[:, 1], That[:, 0])
            minPhi = np.percentile(phi, alpha)
            maxPhi = np.percentile(phi, 100 - alpha)

            vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

            if vMin[0] > vMax[0]:
                HE = np.array((vMin[:, 0], vMax[:, 0])).T
            else:
                HE = np.array((vMax[:, 0], vMin[:, 0])).T

            Y = np.reshape(OD, (-1, 3)).T

            ######## Step 7: Determine concentrations of individual stains ########
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
            maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
            tmp = np.divide(maxC, maxCRef)
            C2 = np.divide(C, tmp[:, np.newaxis])

            ######## Step 8: Convert back to RGB (H&E normalized image) ########
            Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
            Inorm[Inorm > 255] = 254
            Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

            ######## Separate H and E components ########
            H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
            H[H > 255] = 254
            H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

            E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
            E[E > 255] = 254
            E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

            ######## Convert Inorm, H, and E back to RGB before saving ########
            Inorm_rgb = cv2.cvtColor(Inorm, cv2.COLOR_BGR2RGB)
            H_rgb = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
            E_rgb = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)

            ######## Save images using original filenames ########
            # output_path_inorm = f"Data/H&E_test/{i}_Inorm.jpg"
            # output_path_H = f"Data/H&E_Train/msi_1/{i}_H.jpg"
            # output_path_E = f"Data/H&E_Train/msi_1/{i}_E.jpg"
            output_dir="Data/H&E_test"
            output_path_inorm = os.path.join(output_dir, filename)  

            cv2.imwrite(output_path_inorm, Inorm_rgb) 
            # cv2.imwrite(output_path_H, H_rgb)
            # cv2.imwrite(output_path_E, E_rgb)

            i += 1
        
        # cv2.imshow("normalised", Inorm)
        # cv2.waitKey(0)
        # cv2.imshow("seperated_H", H)
        # cv2.waitKey(0)
        # cv2.imshow("seperated_E", E)
        # cv2.waitKey(0)

    j = 0 
    for tile in os.listdir(false):
        tile = cv2.imread(f"{false}/{tile}")
        if tile is None:
            print("hi")
            continue
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        ######## Step 1: Convert RGB to OD ###################
        ## reference H&E OD matrix.
        HERef = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])
        ### reference maximum stain concentrations for H&E
        maxCRef = np.array([1.9705, 1.0308])
        h, w, c = rgb.shape
        rgb = rgb.reshape(((-1, 3)))
        OD = -np.log10((rgb.astype(float)+1)/Io) 

        ############ Step 2: Remove data with OD intensity less than β ############
        # remove transparent pixels (clear region with no tissue)
        ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta

        ############# Step 3: Calculate SVD on the OD tuples ######################
        #Estimate covariance matrix of ODhat (transposed)
        # and then compute eigen values & eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        ######## Step 4: Create plane from the SVD directions with two largest values ######
        #project on the plane spanned by the eigenvectors corresponding to the two 
        # largest eigenvalues    
        That = ODhat.dot(eigvecs[:,1:3]) #Dot product

        ############### Step 5: Project data onto the plane, and normalize to unit length ###########
        ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
        #find the min and max vectors and project back to OD space
        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)


        # a heuristic to make the vector corresponding to hematoxylin first and the 
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:    
            HE = np.array((vMin[:,0], vMax[:,0])).T
            
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T


        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])

        ###### Step 8: Convert extreme values back to OD space
        # recreate the normalized image using reference mixing matrix 

        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

        # Separating H and E components

        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

        ######## Convert Inorm, H, and E back to RGB before saving ########
        Inorm_rgb = cv2.cvtColor(Inorm, cv2.COLOR_BGR2RGB)
        H_rgb = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
        E_rgb = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)

        output_path_inorm = f"Data/H&E_Train/msi_0/{j}_Inorm.jpg"
        # output_path_H = f"Data/H&E_Train/msi_0/{j}_H.jpg"
        # output_path_E = f"Data/H&E_Train/msi_0/{j}_E.jpg"
        cv2.imwrite(output_path_inorm, Inorm_rgb)
        cv2.imwrite(output_path_H, H_rgb)
        cv2.imwrite(output_path_E, E_rgb)
        j += 1
        
        # cv2.imshow("normalised", Inorm)
        # cv2.waitKey(0)
        # cv2.imshow("seperated_H", H)
        # cv2.waitKey(0)
        # cv2.imshow("seperated_E", E)
        # cv2.waitKey(0)


def main():
    directory_true = "Data/SPLIT_IMAGES"
    directory_false = "Data/train_crc/msi_0"
    normalise(directory_true, directory_false)



        

if __name__ == "__main__":
    main()