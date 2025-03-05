// Global objects
var avatarSynthesizer
var peerConnection
var previousAnimationFrameTimestamp = 0;

// Hardcoded Azure Speech parameters
const talkingAvatarStyleName = ""; // Avatar Style (empty)
const usePrivateEndpoint = false; // Enable Private Endpoint is false
const privateEndpointUrl = "";    // Private Endpoint URL (not used since usePrivateEndpoint is false)

// Set additional avatar configurations 
const isCustomAvatar = true;  // Custom Avatar is true
const transparentBackground = true;  // Transparent Background is false
const videoCrop = true;  // Enable video cropping to achieve portrait mode
const backgroundColor = "#00FF00FF"; 

// **Portrait Mode Crop Settings**
// The original video is 1920x1080 (16:9).
// We'll crop horizontally to create a portrait aspect ratio (9:16) for a consistent portrait mode view.
// For a 9:16 portrait ratio with height 1080, the width should be (1080 * 9/16) = 607.5. We will round this to 608 for simplicity.
// Slightly widen the crop
const targetPortraitWidth = 700; // Increased from 608

// Adjust crop calculation to center the wider crop
const cropLeft = Math.floor((1920 - targetPortraitWidth) / 2);
const cropRight = cropLeft + targetPortraitWidth;

// Add at top with other global variables
var isAvatarSpeakingEnded = false;
const followuptimer = 5000;

// Declare variables in the global scope
let azureSpeechRegion;
let azureSpeechSubscriptionKey;
let ttsVoiceName;
let talkingAvatarCharacterName;
let personalVoiceSpeakerProfileID;
let customVoiceEndpointId;
let customVendpoinIDt;

async function fetchConfig() {
    try {
        const response = await fetch('/config');
        const config = await response.json();

        // Use the values dynamically
        // Assign values to global variables
        azureSpeechRegion = config.azureSpeechRegion;
        azureSpeechSubscriptionKey = config.azureSpeechSubscriptionKey;
        ttsVoiceName = config.ttsVoiceName;
        talkingAvatarCharacterName = config.talkingAvatarCharacterName;
        personalVoiceSpeakerProfileID = config.personalVoiceSpeakerProfileID;
        customVoiceEndpointId = config.customVoiceEndpointId;
        customVendpoinIDt = config.customVendpoinIDt;

        //console.log("Config loaded:", config); // Debugging purpose
    } catch (error) {
        console.error("Error fetching config:", error);
    }
}

// Call the function to load config
fetchConfig();


// Setup logging
const log = msg => {
    document.getElementById('logging').innerHTML += msg + '<br>'
}

// Setup WebRTC
function setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential) {
    // Create WebRTC peer connection
    peerConnection = new RTCPeerConnection({
        iceServers: [{
            urls: [iceServerUrl],
            username: iceServerUsername,
            credential: iceServerCredential
        }]
    })

    peerConnection.ontrack = function (event) {
        const remoteVideoDiv = document.getElementById('remoteVideo');
        console.log("Track received:", event.track.kind);
        
        // Clean up existing media elements
        for (let i = 0; i < remoteVideoDiv.childNodes.length; i++) {
            if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
                remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i]);
            }
        }
    
        const mediaPlayer = document.createElement(event.track.kind);
        mediaPlayer.id = event.track.kind;  // Set ID to 'video' or 'audio'
        mediaPlayer.srcObject = event.streams[0];
        mediaPlayer.autoplay = true;
        mediaPlayer.style.objectFit = "cover";
        mediaPlayer.style.width = "100%";
        mediaPlayer.style.height = "100%";
        remoteVideoDiv.appendChild(mediaPlayer);
        console.log(`${event.track.kind} element created with ID: ${mediaPlayer.id}`);
    
        // Hide labels and show overlay
        const videoLabel = document.getElementById('videoLabel');
        if (videoLabel) videoLabel.hidden = true;
        const overlayArea = document.getElementById('overlayArea');
        if (overlayArea) overlayArea.hidden = false;
    
        if (event.track.kind === 'video') {
            mediaPlayer.playsInline = true;
            
            // Get canvas elements and make sure IDs are correct
            const mainCanvas = document.getElementById('mainCanvas') || document.getElementById('canvas');
            const tmpCanvas = document.getElementById('tmpCanvas');
            
            console.log("Canvas elements:", {
                mainCanvas: mainCanvas ? "Found" : "Not found",
                tmpCanvas: tmpCanvas ? "Found" : "Not found"
            });
            
            if (transparentBackground) {
                console.log("Transparent background enabled");
                // Make sure we can see the videos initially for debugging
                remoteVideoDiv.style.visibility = 'visible';
                remoteVideoDiv.style.width = 'auto';
                
                if (mainCanvas) {
                    mainCanvas.hidden = false;
                    mainCanvas.getContext('2d').clearRect(0, 0, mainCanvas.width, mainCanvas.height);
                    console.log("Main canvas is now visible");
                } else {
                    console.error("Main canvas element not found!");
                }
                
                // Add event listener for when video starts playing
                mediaPlayer.addEventListener('play', () => {
                    console.log("Video playback started, initializing transparency");
                    window.requestAnimationFrame(makeBackgroundTransparent);
                });
            } else {
                console.log("Using solid background");
                if (mainCanvas) mainCanvas.hidden = true;
            }
        } else {
            // This is for audio
            console.log("Audio track received");
            mediaPlayer.muted = true;
            mediaPlayer.addEventListener('canplaythrough', () => {
                console.log("Audio can play through, unmuting");
                mediaPlayer.muted = false;
            });
        }
    };
    
    // Update the web page when the connection state changes
    peerConnection.oniceconnectionstatechange = e => {
        log("WebRTC status: " + peerConnection.iceConnectionState);
    
        const stopSessionButton = document.getElementById('stopSession');
        const speakButton = document.getElementById('speak');
        const stopSpeakingButton = document.getElementById('stopSpeaking');
        
        if (peerConnection.iceConnectionState === 'connected') {
            if (stopSessionButton) stopSessionButton.disabled = false;
            if (speakButton) speakButton.disabled = false;
        }
    
        if (peerConnection.iceConnectionState === 'disconnected' || peerConnection.iceConnectionState === 'failed') {
            if (speakButton) speakButton.disabled = true;
            if (stopSpeakingButton) stopSpeakingButton.disabled = true;
            if (stopSessionButton) stopSessionButton.disabled = true;
        }
    };
    

    // Offer to receive 1 audio, and 1 video track
    peerConnection.addTransceiver('video', { direction: 'sendrecv' })
    peerConnection.addTransceiver('audio', { direction: 'sendrecv' })

    // Start avatar, establish WebRTC connection
    avatarSynthesizer.startAvatarAsync(peerConnection).then((r) => {
        if (r.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
            console.log("[" + (new Date()).toISOString() + "] Avatar started. Result ID: " + r.resultId)
            hideLoadingOverlay(); // Hide loading overlay when avatar is ready
        } else {
            console.log("[" + (new Date()).toISOString() + "] Unable to start avatar. Result ID: " + r.resultId)
            if (r.reason === SpeechSDK.ResultReason.Canceled) {
                let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(r)
                if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
                    console.log(cancellationDetails.errorDetails)
                };
                log("Unable to start avatar: " + cancellationDetails.errorDetails);
            }
        }
    }).catch(
        (error) => {
            console.log("[" + (new Date()).toISOString() + "] Avatar failed to start. Error: " + error)
        }
    );
}

function initializeSpeechSDK() {
    if (typeof SpeechSDK !== 'undefined') {
        console.log("Speech SDK successfully loaded.");
    } else {
        console.error("Speech SDK not loaded. Ensure the script is correctly included.");
    }
}

// Function to start the loading bar animation
function startLoadingBar() {
    document.getElementById('downloadBtn').style.display = 'none';   // To hide
    const loadingBar = document.getElementById("loadingProgress");
    let progress = 0;

    // Simulate the loading bar progression
    const interval = setInterval(() => {
        progress += 1.5; // Increment progress
        loadingBar.style.width = `${progress}%`;

        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 100); // Adjust the speed of loading bar progression
}


// Function to wait until the loading bar is full before removing the overlay
async function hideLoadingOverlay() {
    const loadingOverlay = document.getElementById("loadingOverlay");
    const loadingBar = document.getElementById("loadingProgress");

    if (loadingOverlay && loadingBar) {
        // Wait until the loading bar reaches 100% width
        await new Promise((resolve) => {
            const checkLoadingProgress = setInterval(() => {
                const currentWidth = parseInt(loadingBar.style.width);
                if (currentWidth >= 100) {
                    clearInterval(checkLoadingProgress);
                    resolve();
                }
            }, 100); // Check progress every 100ms
        });

        // Once loading is complete, remove the overlay
        loadingOverlay.remove();
        console.log("Loading overlay removed.");
    }
}

// Configuration mapping for avatars
const AVATAR_CONFIG = {
    'dr-david-avenetti': {
        name: 'drdavid-professional',
        style: '',
        customized: true,
        voice: 'drdavidNeural'
    },
    'prof-zalake': {
        name: 'professor-business',
        style: '',
        customized: true,
        voice: 'en-US-EricNeural'
    },
    'lisa-casual': {
        name: 'Lisa',
        style: 'casual-sitting',
        customized: false,
        voice: 'en-US-AvaMultilingualNeural'
    },
    'max-business': {
        name: 'Max',
        style: 'business',
        customized: false,
        voice: 'en-US-BrandonNeural'
    }
};

// Function to get URL parameters
function getUrlParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    const avatarName = urlParams.get('version');
    
    // Debug: Print all URL parameters
    console.log("URL Parameters:");
    urlParams.forEach((value, key) => {
        console.log(`  ${key}: ${value}`);
    });
    
    // Extract parameters
    const params = {
        categoryId: urlParams.get('categoryId'),
        version: urlParams.get('name'),
        name: avatarName,
        description: urlParams.get('description'),
        background: urlParams.get('background'),
        backgroundImage: urlParams.get('backgroundImage') // Check for both parameter names
    };
    
    // Debug: Print the extracted parameters
    console.log("Extracted Parameters:", params);
    
    return params;
}


  
  // Function to set up the avatar with transparent background
  function setupTransparentAvatar() {
    const params = getUrlParameters();
    
    // Get background image from URL parameters
    const backgroundImage = params.backgroundImage || params.background;
    
    // Set the website background
    if (backgroundImage) {
      changeBackground(backgroundImage);
    }
    
  }


// Make video background transparent by matting (currently unused)
// JS: Update makeBackgroundTransparent function
function makeBackgroundTransparent(timestamp) {
    // Get video element
    const video = document.getElementById('video');
    if (!video || video.videoWidth === 0) {
        window.requestAnimationFrame(makeBackgroundTransparent);
        return;
    }
    
    // Get canvas elements
    const mainCanvas = document.getElementById('mainCanvas');
    const tmpCanvas = document.getElementById('tmpCanvas');
    
    if (!mainCanvas || !tmpCanvas) {
        console.error("Canvas elements not found");
        window.requestAnimationFrame(makeBackgroundTransparent);
        return;
    }
    
    // Ensure canvas sizes match video
    if (tmpCanvas.width !== video.videoWidth || tmpCanvas.height !== video.videoHeight) {
        console.log("Adjusting canvas size to match video:", video.videoWidth, "x", video.videoHeight);
        tmpCanvas.width = video.videoWidth;
        tmpCanvas.height = video.videoHeight;
        mainCanvas.width = video.videoWidth;
        mainCanvas.height = video.videoHeight;
    }
    
    // Draw video frame to temporary canvas
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(video, 0, 0);
    
    // Process the frame with chroma key
    const imageData = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height);
    const data = imageData.data;
    
    // This is more aggressive chroma key than before
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        
        // Check if pixel is green (adjust these values for better results)
        if (g > 100 && g > 1.5 * r && g > 1.5 * b) {
            data[i + 3] = 0; // Set alpha to 0 (transparent)
        }
    }
    
    // Draw processed frame to main canvas
    const mainCtx = mainCanvas.getContext('2d');
    mainCtx.putImageData(imageData, 0, 0);
    
    // Hide the video element now that we're showing the canvas
    const remoteVideoDiv = document.getElementById('remoteVideo');
    remoteVideoDiv.style.width = '0.1px';
    
    // Continue the animation loop
    window.requestAnimationFrame(makeBackgroundTransparent);
}

// Modify your setBackgroundImage function to better debug
function setBackgroundImage() {
    const params = getUrlParameters();
    console.log("Setting background image with parameters:", params);
    
    // Determine which background parameter to use (background or backgroundImage)
    const backgroundParam = params.backgroundImage || params.background;
    console.log("Selected background parameter:", backgroundParam);
    
    if (backgroundParam) {
        // Ensure the background parameter is handled correctly
        let backgroundUrl = backgroundParam;
        
        // If it doesn't end with an image extension, add .jpg
        if (!backgroundUrl.match(/\.(jpg|jpeg|png|gif)$/i)) {
            backgroundUrl += '.jpg';
        }
        
        const fullUrl = `/static/images/${backgroundUrl}`;
        console.log("Setting background to:", fullUrl);
        
        document.body.style.backgroundImage = `url('${fullUrl}')`;
        document.body.style.backgroundSize = 'cover';
        document.body.style.backgroundPosition = 'center';
        document.body.style.backgroundRepeat = 'no-repeat';
    } else {
        // Default background if none provided
        console.log("No background parameter found, using default");
        document.body.style.backgroundImage = "url('/static/images/home.jpg')";
    }
}

function startSessionAutomatically() {

    startLoadingBar();
    generateWelcomeButton(); 
    setBackgroundImage();

    const params = getUrlParameters();
    const avatarConfigg = AVATAR_CONFIG[params.name] || {
        name: params.name,
        style: "",
        customized: false
    };
    console.log("Avatar Config: ", avatarConfigg.name, avatarConfigg.style, avatarConfigg.customized);


    // Retrieve session ID from the meta tag
    // Retrieve session ID and user ID from meta tags
    const sessionId = document.querySelector('meta[name="session_id"]').getAttribute('content');
    const userId = document.querySelector('meta[name="user_id"]').getAttribute('content');

    if (sessionId) {
        console.log(`[DEBUG] Session ID: ${sessionId}`);
    } else {
        console.error("[ERROR] Session ID is missing!");
    }

    if (userId) {
        console.log(`[DEBUG] User ID: ${userId}`);
    } else {
        console.error("[ERROR] User ID is missing!");
    }

    // Append the session ID to the URL if necessary
    const currentUrl = new URL(window.location.href);
    if (sessionId && !currentUrl.searchParams.has('session_id')) {
        currentUrl.searchParams.set('session_id', sessionId);
        window.history.replaceState({}, '', currentUrl.toString());
        console.log(`[DEBUG] Session ID appended to URL: ${currentUrl}`);
    }
    

    let speechSynthesisConfig;

    if (usePrivateEndpoint && privateEndpointUrl !== "") {
        speechSynthesisConfig = SpeechSDK.SpeechConfig.fromEndpoint(
            new URL(`wss://${privateEndpointUrl}/tts/cognitiveservices/websocket/v1?enableTalkingAvatar=true`),
            azureSpeechSubscriptionKey
        );
    } else {
        speechSynthesisConfig = SpeechSDK.SpeechConfig.fromSubscription(azureSpeechSubscriptionKey, azureSpeechRegion);
    }

    // Configure for personal voice
    if(avatarConfigg.name == "drdavid-professional"){
        speechSynthesisConfig.endpointId = customVendpoinIDt;
    } else {
        speechSynthesisConfig.speechSynthesisVoiceName = avatarConfigg.voice;
    }
    // Don't set endpointId when using personal voice
    // speechSynthesisConfig.endpointId = customVoiceEndpointId;

    const videoFormat = new SpeechSDK.AvatarVideoFormat();
    if (videoCrop) {
        const cropLeft = Math.floor((1920 - targetPortraitWidth) / 2);
        const cropRight = cropLeft + targetPortraitWidth;

        videoFormat.setCropRange(
            new SpeechSDK.Coordinate(cropLeft, 0),
            new SpeechSDK.Coordinate(cropRight, 1080)
        );
    }
    // Set the format to mp4 and codec to h264
    videoFormat.videoFormat = "mp4";
    videoFormat.videoCodec = "h264";

    const avatarConfig = new SpeechSDK.AvatarConfig(avatarConfigg.name, avatarConfigg.style, videoFormat);
    avatarConfig.customized = avatarConfigg.customized;
    avatarConfig.backgroundColor = backgroundColor;
    
    avatarSynthesizer = new SpeechSDK.AvatarSynthesizer(speechSynthesisConfig, avatarConfig);

    // Log configuration for debugging
    console.log("Avatar configuration:", {
        name: avatarConfigg.name,
        style: avatarConfigg.style,
        customized: avatarConfigg.customized,
        backgroundColor: backgroundColor,
        videoFormat: videoFormat.videoFormat,
        videoCodec: videoFormat.videoCodec,
        transparentBackground: transparentBackground
    });
                
    // WebRTC setup
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `https://${azureSpeechRegion}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1`);
    xhr.setRequestHeader("Ocp-Apim-Subscription-Key", azureSpeechSubscriptionKey);
    xhr.addEventListener("readystatechange", function () {
        if (this.readyState === 4 && this.status === 200) {
            const responseData = JSON.parse(this.responseText);
            const iceServerUrl = responseData.Urls[0];
            const iceServerUsername = responseData.Username;
            const iceServerCredential = responseData.Password;
            setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential);
        } else if (this.readyState === 4 && this.status !== 200) {
            log(`Error fetching the token: ${this.status} ${this.statusText}`);
        }
    });
    xhr.send();

    avatarSynthesizer.avatarEventReceived = function (s, e) {
        console.log(`[Event Received]: ${e.description}`);
        const followUpContainer = document.getElementById('follow_up_questions');
    
        switch (e.description) {
            case "SwitchToIdle":
                console.log("[Event Received]: SwitchToIdle - Removing existing buttons.");
                // Store current questions and clear buttons
                const currentQuestions = Array.from(followUpContainer.children).map(button => button.innerText);
                window.previousFollowUpQuestions = currentQuestions;
                followUpContainer.innerHTML = ''; // Remove buttons
                console.log("[SwitchToIdle]: Stored questions:", currentQuestions);
                break;

            case "TurnStart":
                console.log("[Event Received]: TurnStart - Fetching and updating follow-up questions.");
                fetch('/get_follow_ups')
                    .then(response => response.json())
                    .then(data => {
                        window.pendingFollowUpQuestions = data.follow_up_questions || [];
                        console.log("[TurnStart]: Updated follow-up questions:", window.pendingFollowUpQuestions);
                        //createFollowUpButtons(window.pendingFollowUpQuestions);
                    })
                    .catch(error => {
                        console.error("[TurnStart]: Error fetching follow-up questions:", error);
                    });
                    hideThinkingBubble() ; // Hide thinking bubble when avatar is ready
                    
                    document.getElementById("query_form").style.visibility = "hidden";  
                    //document.getElementById("follow_up_questions").style.visibility = "hidden"; 
                break;
    
                case "TurnEnd":
                    console.log("[Event Received]: TurnEnd - Avatar finished speaking.");
                    // Fetch follow-up questions if not already available
                    // Make visible again
                    RemoveCaption();
                    document.getElementById("query_form").style.visibility = "visible";
                    if (!window.pendingFollowUpQuestions || window.pendingFollowUpQuestions.length === 0) {
                        console.log("[TurnEnd]: Fetching follow-up questions from server.");
                        fetch('/get_follow_ups')
                            .then(response => response.json())
                            .then(data => {
                                window.pendingFollowUpQuestions = data.follow_up_questions || [];
                                const allQuestions = (window.previousFollowUpQuestions || []).concat(window.pendingFollowUpQuestions);
                                console.log("[TurnEnd]: Creating buttons for all questions:", allQuestions);
                                createFollowUpButtons(allQuestions);
                            })
                            .catch(error => {
                                console.error("[TurnEnd]: Error fetching follow-up questions:", error);
                            });
                    } else {
                        const allQuestions = (window.previousFollowUpQuestions || []).concat(window.pendingFollowUpQuestions);
                        console.log("[TurnEnd]: Creating buttons for all questions:", allQuestions);

                        //createFollowUpButtons(allQuestions);
                    }
                    break;
                
            default:
                console.log(`[Unhandled Event]: ${e.description}`);
                break;
        }
    };
}


function cleanupResponseText(responseText) {
    if (!responseText) return '';
    
    // Remove newlines and replace with spaces
    let cleanText = responseText.replace(/\n/g, ' ');
    
    // Remove "Response:" with any number of asterisks around it
    cleanText = cleanText.replace(/\*+Response:\*+\s*/gi, '');
    
    // Remove "Follow-up Question:" with any number of asterisks around it
    cleanText = cleanText.replace(/\*+Follow-up Question:\*+\s*/gi, '');
    
    // Remove any remaining asterisks
    cleanText = cleanText.replace(/\*/g, '');
    
    // Remove any HTML and CSS content
    cleanText = cleanText.replace(/<style>[\s\S]*?<\/style>/gi, '');
    cleanText = cleanText.replace(/<span.*?>(.*?)<\/span>/gi, '');
    
    // Remove special citation text with "[Source]"
    cleanText = cleanText.replace(/\[Source\]/gi, '');
    
    // Remove double spaces that might have been created
    cleanText = cleanText.replace(/\s+/g, ' ');
    
    // Trim any leading/trailing whitespace
    cleanText = cleanText.trim();
    
    return cleanText;
}

function originalSpeakFunction(responseText) {
    document.getElementById('query_form').disabled = true;
    document.getElementById('stopSpeaking').disabled = false;
    document.getElementById('audio').muted = false;

    const params = getUrlParameters();

    const avatarConfigg = AVATAR_CONFIG[params.name] || {
        name: params.name,
        style: "",
        customized: false
    };
    
    // Extract speakable content - now using the return value from showCaption
    const cleanText = showCaption(responseText);
    
    // Generate SSML for speech - using only the clean text
    const spokenSsml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='${avatarConfigg.voice}'>
        <mstts:ttsembedding speakerProfileId=''>
            <mstts:leadingsilence-exact value='0'/>${htmlEncode(cleanText)}
        </mstts:ttsembedding>
    </voice>
    </speak>`;
    console.log("[" + (new Date()).toISOString() + "] Speak request sent.");
    
    avatarSynthesizer.speakSsmlAsync(spokenSsml).then(
        (result) => {
            document.getElementById('query_form').disabled = false;
            document.getElementById('stopSpeaking').disabled = true;
            if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
                console.log("[" + (new Date()).toISOString() + "] Speech synthesized. Result ID: " + result.resultId);
                // Only show follow-up buttons after speech is completed
                isAvatarSpeakingEnded = false;
            } else {
                console.log("[" + (new Date()).toISOString() + "] Speech failed. Result ID: " + result.resultId);
                if (result.reason === SpeechSDK.ResultReason.Canceled) {
                    console.log(SpeechSDK.CancellationDetails.fromResult(result).errorDetails);
                }
            }
        }
    ).catch(log);
}


function generateWelcomeButton() {
    const followUpContainer = document.getElementById("follow_up_questions");
    const queryWrapper = document.getElementById("query_wrapper");
    const queryForm = document.getElementById("query_form");

    followUpContainer.innerHTML = ""; // Clear existing buttons

    // Initially hide the query form
    queryForm.style.display = "none";

    const button = document.createElement("button");
    button.innerText = "Hello!";
    button.onclick = () => {
        // Make invisible (space is still preserved)
        document.getElementById("query_form").style.visibility = "hidden";
        document.getElementById("user_query").value = "Hi, Introduce yourself and explain what can you help me with? what is your knowledge?"
        submitQuery("Hi, Introduce yourself and explain what can you help me with? what is your knowledge?"); // Trigger the query submission
        showThinkingBubble();
        // Show the query form when the initial button is clicked
        queryForm.style.display = "flex";

        // Remove the initial welcome button
        followUpContainer.innerHTML = "";

        // Ensure follow-up container remains visible
        followUpContainer.style.display = "flex";

        // Show the query wrapper if it exists
        if (queryWrapper) {
            queryWrapper.style.display = "block";
        }
    };
    followUpContainer.appendChild(button);

    // Ensure visibility and positioning
    followUpContainer.style.display = "flex";
    console.log("Welcome button added.");
}

function showThinkingBubble() {
    
    document.getElementById('downloadBtn').style.display = 'none';   // To show
    // Find the remoteVideo container
    const remoteVideo = document.getElementById('remoteVideo');

    // Check if the bubble already exists to avoid duplicates
    if (document.getElementById('thinkingOverlay')) return;

    // Create the thinking overlay dynamically
    const thinkingOverlay = document.createElement('div');
    thinkingOverlay.id = 'thinkingOverlay';
    thinkingOverlay.style = `
        position: absolute;
        bottom: -160px;
        left: 45%;           /* Center horizontally */
        transform: translateX(-45%) scale(0.5);  /* Center the element itself and maintain scale */
        z-index: 100;
        pointer-events: none;
        opacity: 0;
        animation: bubbleEnter 0.5s ease-out forwards;
`;
    thinkingOverlay.innerHTML = `
        <img src="/static/images/load.gif" alt="Thinking..." class="thinking-bubble" style="width: 250px; height: auto;">
    `;

    // Add the CSS animations to the document
    const style = document.createElement('style');
    style.textContent = `
@keyframes bubbleEnter {
    0% {
        opacity: 0;
        transform: translateX(-45%) scale(0.5);
    }
    100% {
        opacity: 1;
        transform: translateX(-45%) scale(1);
    }
}

@keyframes bubbleExit {
    0% {
        opacity: 1;
        transform: translateX(-45%) scale(1);
    }
    100% {
        opacity: 0;
        transform: translateX(-45%) scale(0.7);
    }
}
    `;
    document.head.appendChild(style);

    // Append the overlay to the remoteVideo container
    remoteVideo.appendChild(thinkingOverlay);

    console.log("Thinking bubble shown with animations.");
}

function hideThinkingBubble() {
    document.getElementById('downloadBtn').style.display = 'flex';   // To show
    const thinkingOverlay = document.getElementById('thinkingOverlay');
    if (thinkingOverlay) {
        // Add the exit animation
        thinkingOverlay.style.animation = 'bubbleExit 0.5s ease-in forwards';
        
        // Remove the element after animation completes
        setTimeout(() => {
            thinkingOverlay.remove();
        }, 500); // Match this with animation duration
    }
}

function hideButtons(){
    const followUpContainer = document.getElementById('follow_up_questions');
    followUpContainer.innerHTML = "";
    document.getElementById("query_form").style.visibility = "hidden";
}

// Modify createFollowUpButtons to initially hide the container
function createFollowUpButtons(questions) {
    console.log("[Function Call]: createFollowUpButtons");
    const followUpContainer = document.getElementById('follow_up_questions');
    
    // Clear existing buttons
    followUpContainer.innerHTML = '';
    console.log("[createFollowUpButtons]: Cleared existing buttons.");

    // Check if there are questions to create buttons for
    if (!questions || questions.length === 0) {
        console.log("[createFollowUpButtons]: No questions provided for buttons.");
        return;
    }

    // Create buttons for each follow-up question
    questions.forEach((question, index) => {
        const button = document.createElement('button');
        button.innerText = question;
        button.onclick = () => {
            console.log(`[Button Clicked]: ${question}`);
            // Make invisible (space is still preserved)
            document.getElementById("query_form").style.visibility = "hidden";
            document.getElementById('user_query').value = question;
            submitQuery(question);
            showThinkingBubble();
            followUpContainer.innerHTML = ""; // clear buttons when clicked
        };
        followUpContainer.appendChild(button);
        console.log('[createFollowUpButtons]: Button created for question ${index + 1}.');
    });

    followUpContainer.style.display = 'flex'; // Ensure the container is visible
    console.log("[createFollowUpButtons]: All buttons added to the container.");
}

function submitQuery(query = null) {
    // Use the passed parameter if available; otherwise, get the value from the text box
    const userQuery = query || document.getElementById('user_query').value;

    if (!userQuery) {
        console.log("[submitQuery]: No user query provided.");
        return;
    }

    // Client-side JavaScript
    console.log(`[submitQuery]: Sending query: ${userQuery}`);
    const params = getUrlParameters();  // Get URL parameters
    console.log("DEBUG Category ID: ", params.categoryId);

    fetch('/main2', {  // Changed to main2 endpoint
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_query: userQuery,
            name: params.name,
            categoryId: params.categoryId
        })
        
    })
    .then(response => response.json())
    .then(data => {
        console.log("[submitQuery]: Response received from server.");
        // Reset pendingFollowUpQuestions with new data
        window.pendingFollowUpQuestions = [];
        
        if (data.response) {
            console.log("[submitQuery]: Starting avatar speech synthesis.");

            //originalSpeakFunction(data.response);
            originalSpeakFunction(userQuery);
            // fetchfetch
            // Fetch updated follow-up questions
            /*
            fetch('/get_follow_ups')
                .then(response => response.json())
                .then(followUpData => {
                    window.pendingFollowUpQuestions = followUpData.follow_up_questions || [];
                    console.log("[submitQuery]: Follow-up questions updated:", window.pendingFollowUpQuestions);
                })
                .catch(error => console.error("[submitQuery]: Error fetching follow-ups:", error));

                // end here
                */
        } else {
            console.log("[submitQuery]: No response from server.");
        }
    })
    .catch(error => {
        console.error(`[submitQuery]: Error: ${error.message}`);
    });
}

// Function to handle follow-up questions
function submitFollowUp(question) {
    document.getElementById('user_query').value = question;
    submitQuery();
}


function RemoveCaption() {
    console.log("Attempting to remove existing caption");
    const existingCaption = document.getElementById('captionOverlay');
    if (existingCaption) {
        existingCaption.remove();
        console.log("Existing caption removed");
    }
}

function showCaption(text) {
    console.log("ShowCaption called with text:", text);
    
    // Remove citation HTML and CSS, predefined AI assistant span, and trailing </span>
    const cleanedText = text
        .replace(/<style>.*?<\/style>/gs, '')  // Remove any inline styles
        .replace(/<span class="citation-container">.*?<\/span>/gs, '')  // Remove citation spans
        .replace(/<\/span>$/s, '')  // Remove trailing </span> tag
        .trim();  // Remove any trailing whitespace
    
    // Get clean text for speech and duration calculation
    const cleanText = cleanupResponseText(cleanedText);
    
    // Calculate duration
    const CHARS_PER_SECOND = 15;
    const MIN_DURATION = 1500;
    const BUFFER_TIME = 500;
    
    const duration = Math.max(
        MIN_DURATION,
        (cleanText.length / CHARS_PER_SECOND) * 1000 + BUFFER_TIME
    );
    console.log("Calculated duration:", duration, "ms");

    // Find the remoteVideo container
    const remoteVideo = document.getElementById('remoteVideo');
    if (!remoteVideo) {
        console.error("RemoteVideo container not found!");
        return cleanText;
    }
    console.log("RemoteVideo container found");
    
    RemoveCaption();

    // Create caption overlay
    const captionOverlay = document.createElement('div');
    captionOverlay.id = 'captionOverlay';
    captionOverlay.style = `
        position: absolute;
        bottom: 40px;           /* Increased to be more visible */
        left: 50%;
        transform: translateX(-50%);
        z-index: 1011;
        background-color: rgba(0, 0, 0, 0.6);  /* Added background opacity */
        color: white;
        padding: 8px 8px;
        border-radius: 4px;
        font-size: 14px;          /* Increased font size */
        font-weight: bold;        /* Made text bold */
        text-align: center;
        max-width: 95%;
        width: 600px;          /* Added fixed minimum width */
        transition: none;
        text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.2);  /* Added text shadow */
    `;

    // Set caption text (cleaned version without citations)
    captionOverlay.textContent = cleanedText;
    console.log("Caption overlay created with cleaned text");

    // Add to video container
    remoteVideo.appendChild(captionOverlay);
    console.log("Caption overlay added to remoteVideo");

    // Remove caption after calculated duration
    setTimeout(() => {
        if (captionOverlay && captionOverlay.parentNode) {
            captionOverlay.remove();
            console.log("Caption removed after timeout");
        }
    }, duration);

    console.log(`Caption setup complete. Duration: ${duration}ms for ${cleanText.length} characters`);
    
    // Return the clean text for speech synthesis
    return cleanText;
}

// Helper function to HTML-encode text
function htmlEncode(text) {
    const entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;'
    };
    return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match]);
}

async function synthesizeAndPlayTTS(text) {
    if (!text) {
        console.error("Text is required for TTS synthesis.");
        return;
    }

    try {
        // Call the Flask TTS endpoint
        const response = await fetch('/synthesize_audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error(`Failed to synthesize speech: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.audio_url) {
            console.log("Synthesized audio URL:", data.audio_url);

            // Play the synthesized audio
            const audio = new Audio(data.audio_url);
            audio.play()
                .then(() => console.log("Audio playback started."))
                .catch((error) => console.error("Error playing audio:", error));
        } else {
            console.error("Audio synthesis failed:", data.error);
        }
    } catch (error) {
        console.error("Error in TTS synthesis:", error);
    }
}


window.speak = (spokenText) => {
    //fetchChatGPTResponse(spokenText);  // Fetch ChatGPT response
    originalSpeakFunction(spokenText);  // Fetch ChatGPT response
};

window.stopSpeaking = () => {
    document.getElementById('stopSpeaking').disabled = true

    avatarSynthesizer.stopSpeakingAsync().then(
        log("[" + (new Date()).toISOString() + "] Stop speaking request sent.")
    ).catch(log);
}

window.stopSession = () => {
    document.getElementById('speak').disabled = true
    document.getElementById('stopSession').disabled = true
    document.getElementById('stopSpeaking').disabled = true
    avatarSynthesizer.close()
}

window.updataTransparentBackground = () => {
    if (document.getElementById('transparentBackground').checked) {
        document.body.background = './image/background.png'
        document.getElementById('backgroundColor').value = '#00FF00FF'
        document.getElementById('backgroundColor').disabled = true
    } else {
        document.body.background = ''
        document.getElementById('backgroundColor').value = '#FFFFFFFF'
        document.getElementById('backgroundColor').disabled = false
    }
}

window.onload = () => {

    // Call the function to load config
    fetchConfig(); 
    // Retrieve session ID and user ID from meta tags
    const sessionId = document.querySelector('meta[name="session_id"]').getAttribute('content');
    const userId = document.querySelector('meta[name="user_id"]').getAttribute('content');

    if (sessionId) {
        console.log(`[DEBUG] Session ID: ${sessionId}`);
    } else {
        console.error("[ERROR] Session ID is missing!");
    }

    if (userId) {
        console.log(`[DEBUG] User ID: ${userId}`);
    } else {
        console.error("[ERROR] User ID is missing!");
    }

    // Start the session with the configured parameters
    startSessionAutomatically();
    console.log("Session started, ready for speech synthesis.");
};
