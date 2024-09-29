'use client';

import React, { useState, useEffect } from 'react';

const TimeDisplay = () => {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit', 
      hour12: true,
      timeZone: 'America/New_York'
    });
  };

  return (
    <p className="mb-4">
      <span className="text-red-500">📌</span> Newark NJ, time is {formatTime(time)}
    </p>
  );
};

export default TimeDisplay;