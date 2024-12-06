// Global objects
var avatarSynthesizer
var peerConnection
var previousAnimationFrameTimestamp = 0;

// Hardcoded Azure Speech parameters
const azureSpeechRegion = "westus2";  // Region
const azureSpeechSubscriptionKey = "c897d534a33b4dd7a31e73026200226b";  // Subscription Key
const ttsVoiceName = "en-US-RogerNeural";  // TTS Voice
const talkingAvatarCharacterName = "professor-business"; // Avatar Character
const talkingAvatarStyleName = ""; // Avatar Style (empty)
const customVoiceEndpointId = "";  // Custom Voice Deployment ID (empty)
const personalVoiceSpeakerProfileID = ""; // Personal Voice Speaker Profile ID (empty)
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
const targetPortraitWidth = 608; 
// Calculate left and right crop based on the target width. The video feed width is 1920, so:
const cropLeft = Math.floor((1920 - targetPortraitWidth) / 2); // This should be ~656
const cropRight = cropLeft + targetPortraitWidth; // ~656 + 608 = 1264

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

    speechSynthesisConfig.endpointId = customVoiceEndpointId;

    const videoFormat = new SpeechSDK.AvatarVideoFormat();
    if (videoCrop) {
        // Dynamically calculate cropping based on target width
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

    avatarSynthesizer.avatarEventReceived = function (s, e) {
        console.log(`[Event Received]: ${e.description}`); // Log every event received
        if (e.description === "AvatarStarted") {
            console.log("Avatar has started successfully.");
    
            generateWelcomeButton(); // Add the welcome button
        }
    };
    

    
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
}

// Generate the "Hi, who are you?" button
function generateWelcomeButton() {
    const followUpContainer = document.getElementById("follow_up_questions");
    followUpContainer.innerHTML = ""; // Clear existing buttons

    const button = document.createElement("button");
    button.innerText = "Hi, who are you?";
    button.onclick = () => {
        document.getElementById("user_query").value = button.innerText;
        submitQuery(); // Trigger the query submission

        // Ensure follow-up container remains visible
        followUpContainer.style.display = "flex";

        // Show the query wrapper
        document.getElementById("query_wrapper").style.display = "block";
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

// Function to process and make the avatar speak
function originalSpeakFunction(responseText) {
    if (!avatarSynthesizer) {
        console.error("AvatarSynthesizer not initialized. Ensure startSessionAutomatically is called.");
        return;
    }

    // Prepare SSML (Speech Synthesis Markup Language) text
    const spokenText = responseText.replace(/\n/g, ' ');
    const spokenSsml = `
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
            <voice name='en-US-RogerNeural'>${htmlEncode(spokenText)}</voice>
        </speak>
    `;

    console.log("Sending text to avatar:", spokenText);

    avatarSynthesizer.speakSsmlAsync(spokenSsml)
        .then(result => {
            console.log("Speech synthesis completed:", result);
        })
        .catch(error => {
            console.error("Speech synthesis error:", error);
        });
}


function submitQuery() {
    const userQuery = document.getElementById('user_query').value;
    if (!userQuery) return; // Do nothing if the input is empty

    fetch('/main', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_query: userQuery })
    })
        .then(response => response.json())
        .then(data => {
            // Debugging API Response
            console.log("API Response:", data);

            // Update chat history
            const chatHistoryDiv = document.getElementById('chatbot_response');
            chatHistoryDiv.innerHTML += `<p><strong>You:</strong> ${userQuery}</p>`;
            chatHistoryDiv.innerHTML += `<p><strong>Avatar:</strong> ${data.response}</p>`;

            // Make the avatar speak the response
            if (data.response) {
                originalSpeakFunction(data.response);
            }

            // Clear input field
            document.getElementById('user_query').value = '';

            // Handle follow-up questions
            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                console.log("Follow-Up Questions:", data.follow_up_questions); // Debugging
                updateFollowUpQuestions(data.follow_up_questions);
            } else {
                console.log("No follow-up questions available."); // Debugging
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
}

function updateFollowUpQuestions(questions) {
    const followUpContainer = document.getElementById('follow_up_questions');
    followUpContainer.innerHTML = ''; // Clear old buttons

    console.log("Generating follow-up buttons...");
    questions.forEach(question => {
        console.log("Creating button for:", question);
        const button = document.createElement('button');
        button.innerText = question;
        button.onclick = () => {
            document.getElementById('user_query').value = question;
            submitQuery(); // Send the follow-up question as a query
        };
        followUpContainer.appendChild(button);
    });

    // Ensure container is visible and positioned
    followUpContainer.style.display = "flex";
    followUpContainer.style.flexWrap = "wrap";
    followUpContainer.style.gap = "10px";
    followUpContainer.style.justifyContent = "center";

    console.log("Follow-up questions container updated and made visible.");
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


  

function originalSpeakFunction(responseText) {
    document.getElementById('query_form').disabled = true;
    document.getElementById('stopSpeaking').disabled = false
    document.getElementById('audio').muted = false

    // Prepare spoken text by replacing \n with a space for SSML
    let spokenText = responseText.replace(/\n/g, ' '); // Replace newline characters with spaces for SSML

    //let spokenText = responseText
    //let ttsVoice = document.getElementById('ttsVoice').value
    let ttsVoice = ttsVoiceName
    //let personalVoiceSpeakerProfileID = document.getElementById('personalVoiceSpeakerProfileID').value
    let personalVoiceSpeakerProfileID = '';

    let spokenSsml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${ttsVoice}'><mstts:ttsembedding speakerProfileId='${personalVoiceSpeakerProfileID}'><mstts:leadingsilence-exact value='0'/>${htmlEncode(spokenText)}</mstts:ttsembedding></voice></speak>`
    console.log("[" + (new Date()).toISOString() + "] Speak request sent.")
    avatarSynthesizer.speakSsmlAsync(spokenSsml).then(
        (result) => {
            document.getElementById('query_form').disabled = false
            document.getElementById('stopSpeaking').disabled = true
            if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
                console.log("[" + (new Date()).toISOString() + "] Speech synthesized to speaker for text [ " + spokenText + " ]. Result ID: " + result.resultId)
            } else {
                console.log("[" + (new Date()).toISOString() + "] Unable to speak text. Result ID: " + result.resultId)
                if (result.reason === SpeechSDK.ResultReason.Canceled) {
                    let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(result)
                    console.log(cancellationDetails.reason)
                    if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
                        console.log(cancellationDetails.errorDetails)
                    }
                }
            }
        }).catch(log);
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
