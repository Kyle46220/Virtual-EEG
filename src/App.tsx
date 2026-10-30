import { Routes, Route } from 'react-router'
import Home from './pages/Home'
import './App.css'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
    </Routes>
  )
}
