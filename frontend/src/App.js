import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Container, TextField, Button, Typography, 
  Box, List, ListItem, Avatar, CircularProgress, Chip
} from '@mui/material';
import { v4 as uuidv4 } from 'uuid';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function ChatInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [maxTokens, setMaxTokens] = useState(200);
  const [sessionId, setSessionId] = useState('');
  const [bertResult, setBertResult] = useState(null);

  useEffect(() => {
    setSessionId(uuidv4());
  }, []);

  const handleSubmit = async () => {
    if (!input.trim()) return;
    setIsLoading(true);
    try {
      const newMessages = [...messages, { role: 'user', content: input }];
      
      const response = await axios.post(`${API_URL}/chat`, {
        session_id: sessionId,
        messages: newMessages,
        max_new_tokens: maxTokens
      });

      setMessages(response.data.messages);
      setBertResult(response.data.bert_result);
      setInput('');
    } catch (error) {
      console.error('Error:', error);
    }
    setIsLoading(false);
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" gutterBottom>
        AI Chat
      </Typography>

      <Box sx={{ border: '1px solid #ddd', borderRadius: 2, height: '60vh', overflowY: 'auto', mb: 2, p: 2 }}>
        <List>
          {messages.map((msg, index) => (
            <ListItem key={index} sx={{ flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', alignItems: 'start', gap: 2 }}>
              <Avatar sx={{ bgcolor: msg.role === 'user' ? '#1976d2' : '#4caf50' }}>{msg.role === 'user' ? 'U' : 'AI'}</Avatar>
              <Box sx={{ maxWidth: '70%', bgcolor: msg.role === 'user' ? '#e3f2fd' : '#e8f5e9', p: 2, borderRadius: 2 }}>
                <Typography variant="body1" whiteSpace="pre-wrap">{msg.content}</Typography>
              </Box>
            </ListItem>
          ))}
          {isLoading && (
            <ListItem sx={{ justifyContent: 'center' }}>
              <CircularProgress />
            </ListItem>
          )}
        </List>
      </Box>
      
      {bertResult !== null && (
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Chip 
            label={bertResult === 1 ? "Success" : "Failure"} 
            color={bertResult === 1 ? "success" : "error"} 
            sx={{ fontSize: '1.2rem', padding: '10px' }}
          />
        </Box>
      )}

      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <TextField fullWidth variant="outlined" label="Type your message" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} />
        <TextField label="Max tokens" type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} sx={{ width: 120 }} />
        <Button variant="contained" onClick={handleSubmit} disabled={isLoading} sx={{ height: 56 }}>
          Send
        </Button>
      </Box>
    </Container>
  );
}

export default ChatInterface;
