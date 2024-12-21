// Global objects
var avatarSynthesizer
var peerConnection
var previousAnimationFrameTimestamp = 0;

// Hardcoded Azure Speech parameters
const azureSpeechRegion = "westus2";  // Region
//const azureSpeechRegion = "eastus";  // Region
const azureSpeechSubscriptionKey = "c897d534a33b4dd7a31e73026200226b";  // Subscription Key
//const azureSpeechSubscriptionKey = "18f978cca70246309254196a93ce34b4";  // Subscription Key
const ttsVoiceName = "en-US-drdavidNeural";  // TTS Voice
const talkingAvatarCharacterName = "drdavid-professional"; // Avatar Character
const talkingAvatarStyleName = ""; // Avatar Style (empty)
const customVoiceEndpointId = "";  // Custom Voice Deployment ID (empty)
const personalVoiceSpeakerProfileID = "6e315503-b996-485a-8bd7-9f22da3d2ecf"; // Personal Voice Speaker Profile ID (empty)
const usePrivateEndpoint = false; // Enable Private Endpoint is false
const privateEndpointUrl = "";    // Private Endpoint URL (not used since usePrivateEndpoint is false)

// Set additional avatar configurations 
const isCustomAvatar = true;  // Custom Avatar is true
const transparentBackground = false;  // Transparent Background is false
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

// Do HTML encoding on given text
function htmlEncode(text) {
    const entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;'
    };

    return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match])
}

function startSessionAutomatically() {
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
    //speechSynthesisConfig.speechSynthesisVoiceName = "DragonLatestNeural";
    speechSynthesisConfig.endpointId = "8485a9ef-8730-4805-96e1-43c276be1d51";
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
    
                case "TurnEnd":
                    console.log("[Event Received]: TurnEnd - Avatar finished speaking.");
                    // Fetch follow-up questions if not already available
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
                        createFollowUpButtons(allQuestions);
                    }
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
        document.getElementById("user_query").value = "Hello, who are you?"
        submitQuery(); // Trigger the query submission

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


function fetchChatGPTResponse(spokenText) {
    fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: spokenText })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }
            return response.json();
        })
        .then(data => {
            console.log("Received data:", data); // Log the received data
            if (data.response) {
                // Remove "Avatar:" prefix if present
                let cleanedResponse = data.response.replace(/^Avatar:\s*/, "");
                
                document.getElementById('apiResponse').value = cleanedResponse;
                
                // Pass the cleaned response text to originalSpeakFunction
                originalSpeakFunction(cleanedResponse);
            } else {
                throw new Error('No response data found or unexpected structure.');
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            alert('Error: ' + error.message);
        });
}

// Modify createFollowUpButtons to initially hide the container
function createFollowUpButtons(questions) {
    console.log("[Function Call]: createFollowUpButtons");
    const followUpContainer = document.getElementById('follow_up_questions');
    followUpContainer.innerHTML = ''; // Clear any existing buttons

    if (!questions || questions.length === 0) {
        console.log("[createFollowUpButtons]: No questions provided.");
        return;
    }

    questions.forEach((question, index) => {
        const button = document.createElement('button');
        button.innerText = question;
        button.onclick = () => {
            console.log(`[Button Clicked]: ${question}`);
            document.getElementById('user_query').value = question;
            submitQuery();
        };
        followUpContainer.appendChild(button);
        console.log(`[createFollowUpButtons]: Button created for question ${index + 1}.`);
    });

    followUpContainer.style.display = 'flex'; // Make sure the container is visible
    console.log("[createFollowUpButtons]: All buttons added to the container.");
}
 
function submitQuery() {
    const userQuery = document.getElementById('user_query').value;
    if (!userQuery) {
        console.log("[submitQuery]: No user query provided.");
        return;
    }

    console.log(`[submitQuery]: Sending query: ${userQuery}`);
    fetch('/main', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_query: userQuery })
    })
    .then(response => response.json())
    .then(data => {
        console.log("[submitQuery]: Response received from server.");
        window.pendingFollowUpQuestions = data.follow_up_questions || [];
        
        if (data.response) {
            console.log("[submitQuery]: Starting avatar speech synthesis.");
            originalSpeakFunction(data.response);
        } else {
            console.log("[submitQuery]: No response from server.");
        }
    })
    .catch(error => {
        console.error(`[submitQuery]: Error: ${error.message}`);
    });
}

function fetchAndPlayTTS(spokenText) {
    fetch('/synthesize_audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: spokenText }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.audio_url) {
                console.log('Audio URL:', data.audio_url);
                playTTS(data.audio_url); // Play the synthesized audio
            } else {
                console.error('TTS synthesis error:', data.error);
            }
        })
        .catch(error => {
            console.error('Error fetching TTS audio:', error);
        });
}

function playTTS(audioUrl) {
    const audioElement = document.getElementById('ttsAudio');
    audioElement.src = audioUrl;
    audioElement.play()
        .then(() => console.log('Audio playback started.'))
        .catch(error => console.error('Error playing TTS audio:', error));
}


// Function to handle follow-up questions
function submitFollowUp(question) {
    document.getElementById('user_query').value = question;
    submitQuery();
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
    startSessionAutomatically();
    console.log("Session started, ready for speech synthesis.");
};