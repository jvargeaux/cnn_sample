
const URL = 'http://127.0.0.1:5000'
// const url = 'http://167.172.139.55/cnn/'


// === FILE ===

const openFileSelect = () => {
  const fileSelect = document.getElementById('fileSelect')
  fileSelect.click()
}

const onSelectFile = async (event) => {
  const file = event.target.files[0]
  console.log(file)

  fetchResults(file)
}



// === CANVAS ===

let context = null
let isDragging = false
const STROKE = 4

const canvasMouseDown = (event) => {
  if (!context) {
    context = document.getElementById('canvas').getContext('2d')
    context.fillStyle = '#000';
    context.fillRect(0, 0, 200, 200)
    context.fillStyle = '#fff';
  }
  isDragging = true
}

const onDrag = (event) => {
  if (!context || !isDragging) return
  const x = event.offsetX;
  const y = event.offsetY;
  context.fillRect(x - STROKE, y - STROKE, STROKE * 2, STROKE * 2)
}

const onTouchDrag = (event) => {
  event.preventDefault()
  const touch = event.touches?.[0] || event.targetTouches?.[0]
  
  if (!context || !isDragging || !touch) return
  
  const rect = document.getElementById('canvas').getBoundingClientRect()
  const x = Math.max(0, touch.pageX - rect.left);
  const y = Math.max(0, touch.pageY - rect.top - window.scrollY);
  
  context.fillRect(x - STROKE, y - STROKE, STROKE * 2, STROKE * 2)
}

const canvasMouseUp = (event) => {
  isDragging = false
}

const clearCanvas = () => {
  if (!context) {
    context = document.getElementById('canvas').getContext('2d')
  }
  context.fillStyle = '#000';
  context.fillRect(0, 0, 280, 280)
  context.fillStyle = '#fff';

  clearResults()
}

const sendDrawing = () => {
  const canvas = document.getElementById('canvas')
  canvas.toBlob(blob => {
    const file = new File([blob], 'blarg.png')
    fetchResults(file)
  })
}



// === RESULTS ===

const fetchResults = async (file) => {
  let formData = new FormData()
  formData.append('file', file)

  const response = await fetch(URL + '/api/digit', {
    method: 'POST',
    mode: 'cors',
    body: formData
  })
  const responseData = await response.json()
  console.log('response', responseData)

  updateResults(responseData)
  const results = document.getElementById('results')
  results.scrollIntoView()
}

const displayPercentage = (percentage) => {
  if (percentage == undefined) {
    return ''
  }
  return (percentage * 100).toFixed(2) + '%'
}

const updateResults = (responseData) => {
  const results = document.getElementById('results')
  results.innerHTML = responseData?.prediction ?? ''

  const class_0 = document.getElementById('class_0')
  const class_1 = document.getElementById('class_1')
  const class_2 = document.getElementById('class_2')
  const class_3 = document.getElementById('class_3')
  const class_4 = document.getElementById('class_4')
  const class_5 = document.getElementById('class_5')
  const class_6 = document.getElementById('class_6')
  const class_7 = document.getElementById('class_7')
  const class_8 = document.getElementById('class_8')
  const class_9 = document.getElementById('class_9')

  class_0.innerHTML = displayPercentage(responseData?.percentages?.[0])
  class_1.innerHTML = displayPercentage(responseData?.percentages?.[1])
  class_2.innerHTML = displayPercentage(responseData?.percentages?.[2])
  class_3.innerHTML = displayPercentage(responseData?.percentages?.[3])
  class_4.innerHTML = displayPercentage(responseData?.percentages?.[4])
  class_5.innerHTML = displayPercentage(responseData?.percentages?.[5])
  class_6.innerHTML = displayPercentage(responseData?.percentages?.[6])
  class_7.innerHTML = displayPercentage(responseData?.percentages?.[7])
  class_8.innerHTML = displayPercentage(responseData?.percentages?.[8])
  class_9.innerHTML = displayPercentage(responseData?.percentages?.[9])
}

const clearResults = () => {
  updateResults({
    prediction: undefined,
    percentages: []
  })
}