import getProjectPostMetadata from '../../../components/getProjectPostMetadata';
import ProjectPostPreview from '../../../components/PojectPostPreview';

const ProjectPage = () => {
  const postMetadata = getProjectPostMetadata();
  
  // Helper function to parse the date string
  const parseDate = (dateStr: string): Date => {
    const [month, year] = dateStr.split('-').map(Number);
    return new Date(year, month - 1); // Month is 0-indexed in JavaScript Date
  };

  // Sort the postMetadata array by date in descending order (newest first)
  const sortedPostMetadata = [...postMetadata].sort((a, b) => {
    const dateA = parseDate(a.date);
    const dateB = parseDate(b.date);
    return dateB.getTime() - dateA.getTime();
  });

  const PostPreviews = sortedPostMetadata.map((post) => (
    <ProjectPostPreview key={post.slug} {...post} />
  ));

  return (
    <div className='min-h-screen'>
      {PostPreviews}
    </div>
  );
}

export default ProjectPage;