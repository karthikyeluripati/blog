'use client';

import React from 'react';
import Typewriter from 'typewriter-effect';

const TypewriterComponent: React.FC = () => {
  return (
    <Typewriter
      options={{
        strings: ['AI Researcher','Software Engineer', 'Data Scientist'],
        autoStart: true,
        loop: true,
      }}
    />
  );
};

export default TypewriterComponent;