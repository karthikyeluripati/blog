// import SocialIcon from '../../components/SocialIcon';
// const ContactPage = () => {

// }
// {/* <p className='mt-4'>
//         If you have an interesting idea, You can contact me at: 
//       </p>
//       <p className='my-2 flex'>
//       sy493[at]njit[dot]edu <SocialIcon href="mailto:sy493@njit.edu" ariaLabel="Email">
//         <svg className="my-1 mx-1"xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
//         <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
//         </svg>
//         </SocialIcon>
//       </p> */}

// export default ContactPage;

// app/contact/page.tsx

const ContactPage = () => {
    return (
      <div className="space-y-6 mt-24">
        <div className="mb-8">
          <p className="text-gray-700">You can contact me at: sy493[at]njit[dot]edu</p>
        </div>
   
        <div>
          <p className="text-gray-700 mb-4">find me on these platforms:</p>
          <div className="flex gap-2 text-gray-500">
            
              <a 
                href="https://www.linkedin.com/in/karthikyeluripati/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                LinkedIn
              </a>
            
            
              <a 
                href="https://twitter.com/Karthik_Ysvk" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                X
              </a>
              <a 
                href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=kpFqwJYAAAAJ" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                Google Scholar
              </a>
            
            
              <a 
                href="https://github.com/karthikyeluripati" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                GitHub
              </a>
            
            
              <a 
                href="https://discordapp.com/users/karthikysvk" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:underline"
              >
                Discord
              </a>
            
          </div>
        </div>
      </div>
    );
   };
   
   export default ContactPage;