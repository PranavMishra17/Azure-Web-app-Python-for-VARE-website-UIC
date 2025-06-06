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
const backgroundColor = "#FFFFFFFF";  // Background Color (fully opaque white)

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

    // Fetch WebRTC video stream and mount it to an HTML video element
    peerConnection.ontrack = function (event) {
        const remoteVideoDiv = document.getElementById('remoteVideo');
        
        // Clean up existing video element if there is any
        for (let i = 0; i < remoteVideoDiv.childNodes.length; i++) {
            if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
                remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i]);
            }
        }
    
        const mediaPlayer = document.createElement(event.track.kind);
        mediaPlayer.id = event.track.kind;
        mediaPlayer.srcObject = event.streams[0];
        mediaPlayer.autoplay = true;
        mediaPlayer.style.objectFit = "cover"; // Ensure video fills the container without distortion
        mediaPlayer.style.width = "100%";      // Adjust to container width
        mediaPlayer.style.height = "100%";     // Adjust to container height
        remoteVideoDiv.appendChild(mediaPlayer);
    
        // Hide labels and show overlay
        const videoLabel = document.getElementById('videoLabel');
        if (videoLabel) videoLabel.hidden = true;
        const overlayArea = document.getElementById('overlayArea');
        if (overlayArea) overlayArea.hidden = false;
    
        if (event.track.kind === 'video') {
            mediaPlayer.playsInline = true;
            const canvas = document.getElementById('canvas');
            if (transparentBackground) {
                remoteVideoDiv.style.width = '0.1px';
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                canvas.hidden = false;
            } else {
                canvas.hidden = true;
            }
        } else {
            // Mute audio to allow autoplay
            mediaPlayer.muted = true;
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


// Make video background transparent by matting (currently unused)
function makeBackgroundTransparent(timestamp) {
    // Throttle the frame rate to 30 FPS to reduce CPU usage
    if (timestamp - previousAnimationFrameTimestamp > 30) {
        const video = document.getElementById('video')
        const tmpCanvas = document.getElementById('tmpCanvas')
        const tmpCanvasContext = tmpCanvas.getContext('2d', { willReadFrequently: true })
        tmpCanvasContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
        if (video.videoWidth > 0) {
            let frame = tmpCanvasContext.getImageData(0, 0, video.videoWidth, video.videoHeight)
            for (let i = 0; i < frame.data.length / 4; i++) {
                let r = frame.data[i * 4 + 0]
                let g = frame.data[i * 4 + 1]
                let b = frame.data[i * 4 + 2]
                if (g - 150 > r + b) {
                    // Set alpha to 0 for pixels that are close to green
                    frame.data[i * 4 + 3] = 0
                } else if (g + g > r + b) {
                    // Reduce green part of the green pixels to avoid green edge issue
                    let adjustment = (g - (r + b) / 2) / 3
                    r += adjustment
                    g -= adjustment * 2
                    b += adjustment
                    frame.data[i * 4 + 0] = r
                    frame.data[i * 4 + 1] = g
                    frame.data[i * 4 + 2] = b
                    // Reduce alpha part for green pixels to make the edge smoother
                    let a = Math.max(0, 255 - adjustment * 4)
                    frame.data[i * 4 + 3] = a
                }
            }

            const canvas = document.getElementById('canvas')
            const canvasContext = canvas.getContext('2d')
            canvasContext.putImageData(frame, 0, 0);
        }

        previousAnimationFrameTimestamp = timestamp
    }

    window.requestAnimationFrame(makeBackgroundTransparent)
}

function startSessionAutomatically() {

    // Start loading bar as avatar starts loading
    startLoadingBar();

    generateWelcomeButton(); 

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
    speechSynthesisConfig.endpointId = customVendpoinIDt;
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

    const avatarConfig = new SpeechSDK.AvatarConfig(talkingAvatarCharacterName, talkingAvatarStyleName, videoFormat);
    avatarConfig.customized = isCustomAvatar;
    avatarConfig.backgroundColor = backgroundColor;
    avatarSynthesizer = new SpeechSDK.AvatarSynthesizer(speechSynthesisConfig, avatarConfig);

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

                    hideThinkingBubble() ; // Hide thinking bubble when avatar is ready
                break;
    
                case "TurnEnd":
                    console.log("[Event Received]: TurnEnd - Avatar finished speaking.");
                    // Fetch follow-up questions if not already available
                    // Make visible again
                    //RemoveCaption();
                    document.getElementById("query_form").style.visibility = "visible";
                    break;
                
    
            default:
                console.log(`[Unhandled Event]: ${e.description}`);
                break;
        }
    };
    
}

function originalSpeakFunction(responseText) {
    document.getElementById('query_form').disabled = true;
    document.getElementById('stopSpeaking').disabled = false;
    document.getElementById('audio').muted = false;

    const spokenText = responseText.replace(/\n/g, ' ');
    
    //showCaption(spokenText);
    const spokenSsml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='drdavidNeural'>
        <mstts:ttsembedding speakerProfileId=''>
            <mstts:leadingsilence-exact value='0'/>${htmlEncode(spokenText)}
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
                
                
                //hideLoadingOverlay(); // Hide loading overlay when avatar is ready
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
        document.getElementById("user_query").value = "Hi, how are you?"
        submitQuery("Hi"); // Trigger the query submission
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

            originalSpeakFunction(userQuery);

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
    
    // Calculate duration
    const CHARS_PER_SECOND = 15;
    const MIN_DURATION = 1500;
    const BUFFER_TIME = 500;
    
    const duration = Math.max(
        MIN_DURATION,
        (text.length / CHARS_PER_SECOND) * 1000 + BUFFER_TIME
    );
    console.log("Calculated duration:", duration, "ms");

    // Find the remoteVideo container
    const remoteVideo = document.getElementById('remoteVideo');
    if (!remoteVideo) {
        console.error("RemoteVideo container not found!");
        return;
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

    captionOverlay.textContent = text;
    console.log("Caption overlay created with text");

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

    console.log(`Caption setup complete. Duration: ${duration}ms for ${text.length} characters`);
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

// Automatically start the session on page load
window.onload = () => {
    // Call the function to load config
    fetchConfig();

    startSessionAutomatically();
    console.log("Session started, ready for speech synthesis.");
};