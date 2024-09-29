import React from 'react';

const HomePage = () => {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">About</h1>
      <p className="text-lg">
        Primarily, I'm an Engineer.
      </p>
      <p className='text=lg text-slate-400'>Interested in Machine Learning, Data Science and <a className='text-green-600'>Iced Matcha.</a> </p>
      <p className='my-2'>Currently, I'm an AI Engineer at <a className='underline' href='https://www.linkedin.com/company/flozdesign/'>Floz</a>, where I lead engineering efforts on building large scale recommendation systems for construction materials, as well as developing multimodal RAG systems that integrate computer vision for layout info retrieval.</p>
      <p>Previously,</p>
      <ul className="list-disc pl-5 space-y-2">
        <li>Graduated with Masters in Data Science from New Jersey Insitute of Technology.</li>
        <li>I was a Software Engineer Intern at NBCUniversal, where I enhanced cloud infrastructure, automated testing for key user flows, conducted API security testing, and built internal tools for <a className='underline'href='https://www.nbcnews.com/'>NBC News</a> to improve application performance.</li>
        <li>I was a Data Science Intern at <a className='underline' href='https://steerus.io/'>STEERus</a>, I developed a scalable web application designed to assess personality traits and talents, helping users gain deeper insights into their strengths. Additionally, I created a unique scoring system that provides personalized results, which I then translated into detailed, easy-to-understand reports to help users visualize their growth and potential.</li>
        <li>During my time at NJIT, I had the opportunity to work with <a href='https://www.prudential.com/' className='underline'>Prudential Insurance</a>, where I helped build their internal infrastructure and created a real-time project monitoring dashboard, streamlining their operations.</li>
      </ul>
      <p className='my-4'>
      If the world had no intriguing research questions or looming global risks, you'd likely find me backpacking and hike around with my <a className='underline' href=''>film camera</a> in hand. I still make an effort to do this whenever I have some time off...</p><p>I love meeting and having interesting conversations with new people, lets connect!
      </p>
      <p className='my-2'>
      email: sy493[at]njit[dot]edu
      </p>
      {/* <div className=" border-2 border-blue-500 rounded-lg p-2 my-2 mx-9 text-blue-800 text-center">
      <p className="font-bold">
        I'm looking for AI/ML roles! Shoot me an email if you are hiring.
      </p>
    </div> */}

        
      
    </div>
  );
};

export default HomePage;