import React from 'react';
import SocialIcon from '../../components/SocialIcon';

const HomePage = () => {
  return (
    <div className='min-h-screen'>
      {/* <h1 className="text-2xl font-bold mb-6">About</h1> */}
      <p className="text-lg">
        Primarily, I'm an Engineer.
      </p>
      <p className='text=lg text-slate-400'>Interested in Machine Learning and Data Science</p>
      <p className='my-2'>Currently, I'm an AI Engineer at <a className='underline' href='https://www.linkedin.com/company/flozdesign/'>Floz</a>, where I lead engineering efforts on building large-scale recommendation systems for construction materials, as well as developing multimodal RAG systems that integrate computer vision for layout info retrieval.</p>
      <p>Previously,</p>
      <ul className="list-disc pl-5 space-y-2">
        <li>Graduated with a Masters in Data Science from New Jersey Insitute of Technology.</li>
        <li>I was a Software Engineer Intern at NBCUniversal, where I enhanced cloud infrastructure, automated testing for key user flows, conducted API security testing, and built internal tools for <a className='underline'href='https://www.nbcnews.com/'>NBC News</a> to improve application performance.</li>
        <li>I was a Data Science Intern at <a className='underline' href='https://steerus.io/'>STEERus</a>, I developed a scalable web application designed to assess personality traits and talents, helping users gain deeper insights into their strengths. Additionally, I created a unique scoring system that provides personalized results, which I then translated into detailed, easy-to-understand reports to help users visualize their growth and potential.</li>
        <li>During my time at NJIT, I had the opportunity to work with <a href='https://www.prudential.com/' className='underline'>Prudential Insurance</a>, where I helped build their internal infrastructure and created a real-time project monitoring dashboard, streamlining their operations.</li>
      </ul>
      {/* <p className='my-4'>
      If the world had no intriguing questions and compeling solutions you'd likely find me backpacking and hike around with my <a className='underline' href=''>film camera</a> in hand. I still make an effort to do this whenever I have some time off...</p> */}
      {/* <p className='mt-4'>
        Publications:
        <ul>
        <li><a href='https://ieeexplore.ieee.org/document/9670956' className='underline mt-2'>Forecasting electrical demand for the residential
        sector at the national level using deep learning.</a></li>
        <li><a className=''>Dharmoju, P.K.,<a className='underline'>Yeluripati, K.</a>, Guduri, J., & Palle, K.</a></li>
        <li>IEEE AIMV, September 2021</li>
        <li><a className='underline' href='https://www.karthikyeluripati.me/projectposts/electrical-demand-forecasting-blog'>Blog-post</a></li>
        </ul>
      </p> */}
      <p className='mt-4'>
        I’m all about connecting with new people and diving into interesting conversations, let’s connect!
      </p>
      <p className='my-2 flex'>
      email: sy493[at]njit[dot]edu <SocialIcon href="mailto:sy493@njit.edu" ariaLabel="Email">
        <svg className="my-1 mx-1"xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
        </svg>
        </SocialIcon>
      </p>

        
      
    </div>
  );
};

export default HomePage;