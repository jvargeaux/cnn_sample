
const getDigit = async () => {
  console.log('getting digit...')

  const data = {
    name: 'hello'
  }

  const response = await fetch('http://127.0.0.1:5000/api/digit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  const responseData = await response.json()
  console.log('response', responseData)
}