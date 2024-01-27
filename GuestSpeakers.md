# Guest speakers BatchelorsProef

## Faktion

- Jeroen Boeye

Faktion,

The software part is just as important as the ai solution.

Don’t neglect the software engineering aspect

- Chatlayer
  - Conversational flows
  - Chat bot
  - Questions about the subject
- Metamaze

  - Automated document processing ( on documents or emails)
  - makes a small summary of the document
  - Supervised ML

- The road to a successful ai project

  - Validate the business case
    - Dont use the latest coolest software, its not always the best to use in this use case
    - Look at the problem that needs to be solved first
    - What is the status today? How good is the human version in comparison with the AI.
    - Determine the evaluation metrics
  - Is the Data there?

    - What if it isn’t there? Game over? —> NO

    - Data quantity vs quality -

  - Fail fast, learn faster
    - Try to implement the difficult or highest risk part early on.
      - If 90% of the project works but the hardest part doesn’t, it was all useless
  - Communication with the customer
    - Communicate the failures

— Use cases: detecting solar panels

- we want to create a list of addresses of houses with. Solar panels
- do you have data?
- no, not data we can share
- Time to get creative!
- geopunt
- look for open data sources
- data engineering
- postgres
- postGIS (geo data)
- clip polygon with addresses to the images with the address
- output -> image id with address and then images on image id
- Use case: of this method: using AI to tax unofficial declared pools

-PCA ?? Look up

- features in the last layer of the NN
- compare fake to real ones and adjust so the PCA of the fake ones are closer to the real ones

- Use case 2: increasing safty and saving time

  - Damage detection application of containers during flights
    - A lot of change of responsibility (different people working on it)
  - We want to automate visual inspection of our air freight containers
    - Do you have data?
      - “Don’t worry we will do it” —> ask Cleary for what you want to have, follow up on the data collection process
      - OCR ( optical character recognition )
      - Time to get creative again!
        - Only 30 pictures and a lot of different types of damages
        - 3d images of the containers (from website
        - Synthetic data generator (make damage in 3d field)
        - Quality vs quantity
          - A lot of quantity suddenly but low quality
        - Only trained on the known methods of damage. So not scalable to now methods of damage.
          - Solution: anomaly detection
            - Don’t train on damage, but on the normal containers.
            - CGAN (conditional generative adversarial network)
              - Generate network (‘the artist’)
                - Add noise to input
              - Critic network (art critic)
                - Determines how real the image is.
                - It gets fed real and fake images occasionally
              - Lock generator and critic
                - Make new network with an encoder
                  - Gets a real image, and tries to create a fake one based on that image
                  - The generator now tries to generate a fully complete image without damage, can be layered of the encoded image to see the generated part

- Use case 3: Detecting time series anamolies in wind turbines

  - We want to detect early warning signals for failures in a new type of gearbox.
    - Do you have any data?
      - Yes but only healthy data
        - Good quantity but medium quality
        - Easy to predict the future, due to high amounts of healthy data
          - Once the data is abnormal, there will be a high reconstruction error
        - Make synthetic anomaly data
      - Fourier Transform on time series data.
        - FFT - Adapt the frequency - When anomaly, double the frequency. - Use inverse of this to generate the synthetic data
      - The data is different now, and anamaly will try to be predicted as a ‘normal’ value. But reconstruction error will be higher.
        - Use the reconstruction error to determine the anomaly instead
        - When reconstruction error fluctuates—> anomaly

- Use case4: Wienerberger Energy Analysis

  - The ovens used to bake cement blocks (bakstenen) uses as much energy in 3 hours as a family in a year
  - We want to reduce energy consumption in our brick oven
    - Do you have any data?
      - Yes but we have 43k csv files
      - Pre-processing took the longest
        - Gas volume /15 min —> api
        - Production data —> api
          - Brick types and weights
          - N bricks per kiln per type
        - Sensor data —> given from customer
          - Diferent formats
          - Values per minute
            - Parse for multiple formats
      - High quantity, bad quality
      - Data is generic, not specific placement of the bricks in which section of the oven
      - In dit ,moment zit er zoveel gewicht van zoveel stenen in deze sectie van de oven
        - How do we get a dataset from this?
        - There was a timetable with each start time of each process
          - Join on other dataset
          - Oven = multiple zones, 104 positions
          - Create digital twin !important!
            - Give me the stones in this zone
        - Now data quality has improved a lot
      - Beheers u data!! Praat met experten
      - Used a simple linear model after all the preprocessing

- Use case 5: Petersime

  - Incubation of eggs
    - Lots of sensor data
    - PCA cycle similarity
  - Datastorage (azure blob)

    - Azure Functions trigger
      - Runs every 24h
        - Pipeline -> azureML
    - Files -> readable json
      - Save in blob storage
    - Parse json —> injest + process
      - Gets saved in SQL meta database
        - Look at last data saved for that sensor
      - Azure data base
    - AZ function -> TFSRESH
    - Azure container apps
      - BICEP for infra as code
      - Backend container
        - Backend gets the sql and azure refresh
      - Frontend container

  - Data - Cycle - Time - Sensor1 - Sensor2 - Sensor…
    - TSFRESH
      - Gets aggrogatic statistics outs of time series data
      - Using PCA

- Use case 6: Sibelco (sand)

  - TNK? ( time series anomaly detection)
  - Theorics compared to real live model —> make virtual twin

- Preprocessing is hard but essential, MASTER IT
- Don’t treat data as a static resource!! You can make changes and alter it
  - We know which models work, lets make data better instead of improving the models
  - Invent it
  - Change the task
- Make a dashboard for easy overview
- Mount the folder you are coding in —> now code automatically gets imported into the containers (HOST MOUNT)

  - Don’t need to build to see changes in some files.

- Streamlti for dashboards

## Noest

- Toon Vanhoutte

- Noest

  - Cronos Group
  - 56 employees
  - local employees
  - Pragmatisch werken (flexibel)
  - global impact
  - 3 pillars
    - vakmanschap
    - parterschap
    - plezier
  - assortiment

    - app
    - cloud
    - data
    - ai
    - low-code
      - power platform, drag and drop (front end)
    - erp
    - integration
    - Mincrosoft partner

  - case 1

    - verpakkingen bedrijf
    - Image search op basis van product labels

      - product op naam
      - objecten en kleuren
      - tekst
      - certificatie logo's
        - wat zijn onze labels, welke labels hebben we nodig voor in dit land (bv andere labels voor andere landen nodig)
      - internal tool voor alle klanten
      - externe tool waarbij elke klant toegang heeft tot zijn eigen images

        - OCR?? (optical character recognition) not good enough

      - Challenges;

        - inconsistent PDF's
          - text als expliciete text
          - tekst als image
        - Grotere PDF's (100 mb)
          - trage processing
          - Dure processing
          - Hoger dan limitations van vele services (bv azure, gpt)
        - Solution:
          - azure blob storage
            - contaner with pdfs
          - ??? (this is the challenge)
            - pdf conversion
            - will take a while
            - must be cheap
            - must be reliable
          - azure blob storage
            - container with images (jpg's)
          - Solution for ???:
            - azure event based
              - azure event grid
                - notification service
            - pdf conversion (beval/pdf2image)
              - must run in docker
            - lang lopend process
              - api call must be fast, if open for too long it will time out
                - not good for production, acceptable for proof of concept
                - solution: api call triggers even --> queue
                  - queue triggers function
            - relaible
              - azure logic apps instead of functions ( functions can be too techincal)
                - logic apps allow to rerun a single step
            - cheap
              - azure container apps
                - easy to scale
                - KEDA (event driven auto scaling)
                  - 5 queues?? --> new instance
                    - descale when queue is empty or near empty
            - End solution for ??:
              - azure blob storage --> event grid --> logica app --> container app that does pdf conversion --> saves in blob storage and send notifictation so the next step can start

      - demo:

        - azure event subscription on blob creation and update
          - returns webhook with the blob url
            - extern messages need to get an accept token or else ddos would be possible (not needed for local dev)
            - call back url (web hook, so you know when the process is done)
            - queue chooses the next message in the queue not visa versa. (for scaling)

      - Chellenge 2

        - images doorzoekbaar maken

          - tekst op images
          - objecten op images

          - azure AI search
            - content --> azure AI search --> index --> search --> query to your app
          - query on text has existed for a long time but not on images??

            - example search for shower gell --> searched the labels of the products (could also be washing gell, or douche gell... needs to be able to find all of these)
            - solution:
              - LLM (large language model)
            - Vectors of similarity of words (expl. queen = king - man + woman)
            - Vector search
              - images, documents, audio --> vecoter representation --> K-nearest neighbour search --> results <-- vector representation (transform into embedding)<-- query
              - very specific embedding models
                - images --> image embedding
                - text --> text embedding
                - audio --> audio embedding
              - QUery needs to be on the same embedding as the data
                - needs to disregard the language the query is in as well as the language the data is in
            - Search index:
              - de zoektabel
                - id
                - name
                - url
                - tenant name
                  - support multiple customers
                  - fecatable --> we can query on this
                  - klanten kunnen inloggen en enkel hun data zien
                - imageVector
                  - SingleCollection (integer array)
                - imageDescription
                  - wat voor label?
                  - welke icoontjes voor de eu bv?
                  - welke certificaten?
                - imgDescriptionVector
                  - SingleCollection
                - imageText
                - imageTextVector
                  - SingleCollection
            - Concepts:
              - data source --> indexer --> index
                - is er nieuwe data? --> indexer --> index
            - Indexer

              - uses skills to extract data
                - Standard skill
                  - OCRSkill
                    - pure text of image
                      - saved into image text index
                  - TextEmbeddingSKill
                    - text --> openai model --> embedding
                      - saved into image text vector index
                - Custom webSkill
                  - GPT4_vision skill
                    - image description with c
                    - into Standard skill
                      - Text embedding skill
                  - Image embedding into gpt4-vision
              - Tenant name??
                - comibnation of ocrskill and gpt4-vision skill
                  - String concatenate --> tenant detection (how oftne is a word used?? map onto tenant name)

            - demo of notebook that creates the whole azure ai search....

              - ocr vs gpt4-vision
                - ocr doesnt translate, and doesnt classify, only directly shows the text detected
                - gpt4-vision translates and classifies the text, easy to navigate through the data
                - ocr doest give the option to search on text only... gpt4-vision gives more a semantic search
                  - tentant could want to search for 1 specific word, but not the other words in the text (only works in rare cases )
                - gpt4-vision is slow... (currently could change in the future)
                - gpt4-vision allows for promts
                  - describe this image... give use the brand name.... give us the product name... give us the product type... give us the nutricional values... give us the ingredients ... give the certificates (if not found, dont give anything, prevent being able to index on something that gives 'not found')
              - text search on ocr vs gpt4
                - barcode works for both
                - colors? only gpt4

      - Challenge 3:

        - Verschillende zoekopties (the lion was heard form all around)

          - keyword search
            - 1. the lion is an apex predator (finds the lion word)
            - 2. a lion roard loudly (finds the lion word)
          - vector search
            - 1. a lion roard loudly (finds the lion word)
            - 2. the king of the jungle made the thundering sound (finds the lion word)
            - 3. the lion is an apex preditor (finds the lion word)
            - 1 & 2 have a more similar vecotor than 3
          - hybrid
            - finds all 3, uses vector search to sort the results after the keyword search
          - semantic ranker
            - re-ranks the results semantic vector search
            - uses context of sematic meaning of a query to rank the search results
            - returns semantic caption and the highlights

        - demo:

          - search on word "groenten"
            - keyword search
              - finds nothing
            - vector search
              - finds all the images with vegetables in them
          - search on word "bloemkool"
            - search on word "bloemkool"
              - keyword search
                - finds nothing
              - vector search
                - finds all the images with vegetables
          - search on 'boontjes van 1kg'

            - keyword search
              - finds nothing
            - vector search
              - finds the right image --> specific enough

          - THis can then be put in gpt4-vision with the original search query and the images returned.
            - searches for the most similar things in the images to the query

      - to customers to:
        - always mention ai was used and the results can be wrong
      - What have we learned:
        - durf uit de comfort zone te stappen, deze technologie is heel nieuw
        - built-in indexer heeft weinig mogelijkheden om te interveneer (ideaal een externe indexer gebruiken)
          - meer controlere zou leuk zijn (momenteel nog niet mogelijk of niet haalbaar)
        - prompt engineering is een noodzaak, maar geen job
        - vector seach en llms zijn geen echte wetenschap
        - de ai techoogie is nog niet volwassen, en evolueert snel
      - klanten gebruiken bijna nooit zelf gpt, zijn vaak verbaasd over de mogelijkheden, mensen zijn vaak sceptisch over gpt of verwachten juist te veel....
